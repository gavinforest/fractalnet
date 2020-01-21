{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeSynonymInstances #-}
{-# LANGUAGE NamedFieldPuns #-}
--{-# LANGUAGE Strict #-}

import Data.Char (chr,ord)
import Data.List (foldl')
import System.Random (mkStdGen, random, randoms)
import System.IO(IOMode(..), hClose, hGetContents, openFile)

import GA (Entity(..), GAConfig(..), 
           evolveVerbose, randomSearch)
import Data.Array.Accelerate as A
import Data.Array.Accelerate.CUDA
import Control.Monad.Random
import Data.List

-- efficient sum
sum' :: (Num a) => [a] -> a
sum' = foldl' (+) 0


--tree expression data type
data Node = Value Float| Var | DBranch ( Float -> Float -> Float) Node Node | SBranch (Float -> Float) Node
    deriving (Show, Eq, Read)

evaluate :: Node -> Float -> Float
evaluate (Value x) var = x
evaluate Var var = var
evaluate (DBranch func node1 node2) var = func (evaluate node1 var) (evaluate node2 var)
evaluate (SBranch func node1) var = func $ evaluate node1 var

data Genome = GenomeConstr {
                                res :: Integer,
                                funcs :: [[Node]]
                                }
                                    
primitives1 = [cos, sin, sqrt, (**2)]
primitives2 = [(/), (+), (-), (*)]

myRound :: Int -> Float -> Float
myRound resol x = let pos =  if x >= 0 then head $ filter (\n -> n >= -1 and n <= 1) $ iterate (-2) x
                                       else head $ filter (\n -> n >= -1 and n <= 1) $ iterate (+2) x
                  in if pos >= 0 then (floor $ pos * resol) / resol
                                 else (ceiling $ pos *resol) / resol



applyFuncs :: [Float] -> Genome -> [[Float]] 
applyFuncs vec genome = let fs = funcs genome
                            r = res genome
                        in concat [[update ind (myround r $ x + fx) vec | f <- fs !! ind ] | ind <- [0..length fs -1]]
                            
applyFuncsMult :: [[Float]] -> Genome -> [[Float]]                                
applyFuncsMult vecs genome= concatMap applyFuncs vecs genome 
                                
locateSample :: Genome -> [Float] -> [[Float]]
locateSample genome sample = --just creates spacial vectors, no activations
    let factors = head [(i, (length sample)/i) | i <- [floor . sqrt $ length sample .. 1], (length sample) `mod` i == 0]
        dimension = length $ funcs genome
        resIt = myRound $ res genome
        in let
            locPairs = [[resit $ a/(fst factors) - 0.5 ,resit $ b/(snd factors) - 0.5] | a <- [1..fst factors], b <- [1..snd factors]]
            empties = replicate (length sample) $ replicate (dimension - 2) 0
            in zipWith (++) locPairs empties


--CONSOLIDATION FUNCTIONS
--Consolidation List format:
--  A consolidation list should be a list of list of sublists, where each sublist contains the set
--  elements that overlap. These will be added to create a new element in their position.
--  if no other elements overlap an element, it will be a list of form [element].


consolGen :: Int -> [[Int]] -> [[Float]] -> [[Int]]
consolGen offset currList _ = currList 
consolGen offset currList (v:vecs) = if filter (elem offset) (concat currList) == [] then
    let indexs = foldl' (\x (i, acc) -> if x == v then (i+1, acc ++ [i]) else (i+1, acc)) (offset, []) vecs
        in consolGen (offset+1) (currList ++ [indexs]) vecs
    else consolGen (offset + 1) currList vecs
    
synapseNumThreshold = 50000
genConsolidate :: Genome -> [Float] -> [[[Int]]]
genConsolidate gene vecSample = 
    let inp = locateSample gene vecSample
        fs = funcs gene
        belowThreshold = \(ouputVec, consols) -> length . concat consols <= synapseNumThreshold
        consolGenOne = consolGen 0 []
        in snd $ last $ filter belowThreshold $ iterate branch (inp, [])
            where branch (vec, cons) = 
                let new = applyFuncsMult inv gene 
                in (floatConsol new (consolGenOne new) , cons ++ [consolGenOne new])

floatConsol :: [[Float]] -> [[[Int]]] -> [[Float]]
floatConsol inp conslist = concatMap getAdd conslist
    where getAdd = foldl' (+) 0 $ map (inp !!)

consol :: Acc (Vector Float) -> [Acc (Vector Int)] -> Acc (Vector Float)   
consol inp conslist= concatMap (getAdd) conslist
    where 
        getAdd = fold (+) 0 $ A.map ((A.!!) inp) --fold is a accelerate function
        
        
--WEIGHT GENERATOR
genWeight :: (RandomGen g) => Int -> Rand g [Float]
genWeight n = sequence $ replicate n $ getRandomR [-1..1]

genWeights :: (RandomGen g) => [[[Float]]] -> Rand g [[Float]]
genWeights consols = fold (\x acc -> acc ++ [genWeight . length . concat x] ) [] consols


--BIAS GENERATOR
genBiases :: (RandomGen g) => [[[Float]]] -> Rand g [[Float]]
genBiases consols = fold (\x acc -> acc ++ [genBias . length . concat x]) [] consols
    where genBias n = sequence $ replicate n $ getRandomR [-2..2]



--FEEDFORWARD AND ERROR METHODS
feedForward :: Acc (Vector Float) -> [[Acc (Vector Int)]] -> Acc (Scalar Int) -> [Acc (Vector Float)] -> [Acc (Vector Float)] -> Acc (Vector Float)
feedForward input consols branching ws bs=  
    fold (\(w,b,cons) (inp,prev) -> 
        let p = A.map (tanh) $ consol $ A.zipWith (*) $ A.zipWith (+) b (A.concatMap (A.replicate branching) inp) w cons 
        in (p, prev ++ [p])) 
        (input, []) zipped
            where zipped = zip3 ws bs consols


l2Error :: Acc (Vector Float) -> Acc (Vector Float) -> Acc Float  
l2Error predicted outp = sqrt $ A.fold (+) 0 $ A.map (**2) 0 $ A.zipWith (-) outp predicted

grad inp outp consols branching ws bs fws fbs = 
    let outputs = snd $ feedForward inp consols branching ws bs
        in let complete = outputs ++ 

alpha = 0.001
descend inp outp consols branching ws bs fws fbs=
    let outputs = snd $ feedForward inp consols branching ws bs
        in let complete = ouputs ++ [A.map tanh $ A.
            in (err, (zipWith $ zipWith $ zipWith (\g val -> val - g * alpha)) $ grads [ws,bs] )

--GENETIC ALGORITHM

--
-- GA TYPE CLASS IMPLEMENTATION
--
maxFuncDepth = 4
maxDimFuncs = 5
maxDims = 20
constantLims = (-5.0, 5.0)
resolutionLimits = (0,100)
resModifyRange = (0,5)

selectRandom list g = (!!) (fst $ randomR (0, length list) g) list

genRandomNode _ seed = Var
genRandomNode 1 seed =  if randomR (0,1) (mkStdGen seed) == 0 then Value $ fst $ randomR constantLims
                                                              else Var
genRandomNode n seed = let (nodetype, g)  = randomR (1,2) $ mkStdGen seed
                       in if nodetype == 1 then SBranch (primitiveone) (genRandomNode (n-1) s1)
                                           else DBranch (primitivetwo) (genRandomNode (n-1) s1) (genRandomNode (n-1) s2)
                          where
                          primitiveone = selectRandom primitives1 g
                          primitivetwo = selectRandom primitives2 g
                          s1 = fst $ random g
                          s2 = fst $ random $ mkStdGen s1
                          
--                                                  make new seed               add random node to list
genRandomDimNodes seed = foldl' (\n (g,list) -> (fst $ random $ mkStdGen g, list ++ [genRandomNode n g])) (seed, []) depths
    where depths = take (fst $ randomR (0, maxDimFuncs) $ mkStdGen seed) $ randomRs (1, maxFuncDepth) (mkStdGen seed)

duplicate l = map (\x -> [x,x]) l

myZipWith l1 l2 = let nl1 = length l1
                      nl2 = length l2
                  in if nl1 < nl2 then (take nl1 $ zipWith (\x y-> [x,y]) l1 l2) ++ duplicate $ drop nl1 l2
                                  else (take nl2 $ zipWith (\x y-> [x,y]) l1 l2) ++ duplicate $ drop nl2 l2
                                  
update index element list = take index list ++ [element] ++ drop (index + 1) list




instance Entity Genome Float Data [Func] IO where
  -- generate a random entity, i.e. a random string
  genRandom primitives seed = return $ Genome {res = resolution, funcs = dimLists}
    where
        g = mkStdGen seed
        (resolution, g1) = randomR resolutionLimits g
        (nDims,g2) = randomR (1,maxDims) g1
        seeds = randoms $ mkStdGen $ fst $ random g2
        
        dimLists = map (\s -> genRandomDimNodes s) $ take nDims seeds

  -- crossover operator: mix (and trim to shortest entity)
  crossover _ _ seed e1 e2 = return $ Just e
    where
      g = mkStdGen seed 
      cps = myZipWith e1 e2
      picks = map (flip mod 2) $ randoms g
      e = zipWith (!!) cps picks

  mutation pool p seed e = return $ Just final
    where
      g = mkStdGen seed
      numMutations = round $ (length $ concat (funcs e)) * p :: Int
      dimsToMutate = sort $ take numMutations $ randomRs (0, length (funcs e) -1) g  --sort might not be imported
      willAdd = (fst $ randomR (0,1) g) == 1
      (aSeed,g1) = random g
      willAddDim = fst (randomR dimAddingRange g) == 0
      (dimToAddTo, g2) = randomR (0, length (funcs e) -1) g
      seeds = randoms g2 :: [Int]
      toModify = take (length dimsToMutate) $ zip dimsToMutate seeds
      
      toInd genSeed list = fst $ randomR (0, length list -1) $ mkStdGen genSeed
      genRand genSeed = genRandomNode (fst $ randomR (1,maxFuncDepth) $ mkStdGen genseed) genSeed
      -- i=dim to mutate, s=random seed
      newNodes =  foldl'(\(i,s) acc -> update i (update (toInd s (acc !! i)) (genRand s) (acc !! i)) acc) (funcs e) toModify  
      
      addMutated = if willAdd 
                     then if willAddDim
                         then newNodes ++ [genRandomDimNodes (fst $ random g1)]
                         else update dimToAdd (newNodes !! dimToadd ++ genRand aSeed) newNodes
                     else if willAddDim
                         then update dimToadd [] newNodes
                         else let ind = toInd aSeed newNodes
                              in update ind (update ( toInd (seeds !! 101) (newNodes !! ind) ) [] newNodes !! ind) newNodes
       final = if randomR resModifyRange g1 == 1 then Genome {res = fst $ randomR resolutionLimits g1, funcs = addMutated}
                                                 else Genome {res = (res e), funcs = addMutated}

  -- NOTE: lower is better
  --dataset is ([trainimages], [trainlabels], [testimages], [testlabels])
  score dataset genome = do
    let
        !cons = map (map (\n -> use $ fromList (Z:.(length n)) n :: Vector Int)) $ genConsolidate genome $ head $ fst dataset
        !ws = map (\n -> use $ fromList (Z:.(length n)) n :: Vector Int) $ evalRandIO $ genWeights consols
        !bs = map (\n -> use $ fromList (Z:.(length n)) n :: Vector Int) $ evalRandIO $ genBiases consols
        !branching = use $ fromList (Z) $ (:[]) $ length . concat $ funcs genome
    
    

  -- whether or not a scored entity is perfect
  isPerfect (_,s) = s == 0.0


main :: IO() 
main = do
        let cfg = GAConfig 
                    100 -- population size
                    25 -- archive size (best entities to keep track of)
                    300 -- maximum number of generations
                    0.8 -- crossover rate (% of entities by crossover)
                    0.2 -- mutation rate (% of entities by mutation)
                    0.0 -- parameter for crossover (not used here)
                    0.2 -- parameter for mutation (% of replaced letters)
                    False -- whether or not to use checkpointing
                    False -- don't rescore archive in each generation

            g = mkStdGen 0 -- random generator

        
        -- Note: if either of the last two arguments is unused, just use () as a value
        es <- evolveVerbose g cfg [] dataSet
        let e = snd $ head es :: String
        
        putStrLn $ "best entity (GA): " ++ (show e)

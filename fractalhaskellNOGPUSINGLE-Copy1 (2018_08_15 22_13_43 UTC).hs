{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeSynonymInstances #-}
{-# LANGUAGE NamedFieldPuns #-}
{-# LANGUAGE BangPatterns #-}
 -- {-# LANGUAGE Strict #-}

import Data.Char (chr,ord)
import Data.List (foldl')
import System.Random (mkStdGen, random, randoms)
import System.IO(IOMode(..), hClose, hGetContents, openFile)

import GA (Entity(..), GAConfig(..), 
           evolveVerbose, randomSearch, Archive)
import Control.Monad.Random
import Data.List as DL
import Data.List.Split
import Codec.Compression.GZip (decompress)
import qualified Data.ByteString.Lazy as BS
import Control.Monad
import Data.Functor
import Data.Ord
-- import Data.Vector as V


-- efficient sum
sum' :: (Num a) => [a] -> a
sum' = foldl' (+) 0


--tree expression data type
data Node = Value Float| Var | DBranch PrimitiveTwo Node Node | SBranch PrimitiveOne Node
    deriving (Show, Eq, Read,Ord)
    
data PrimitiveOne = Cos | Sin | Sqrt | Square
    deriving (Show, Eq, Read, Ord)
data PrimitiveTwo = Add | Sub | Mult
    deriving (Show, Eq, Read, Ord)
    
toFunc1 :: PrimitiveOne -> (Float -> Float)
toFunc1 Cos = cos
toFunc1 Sin = sin
toFunc1 Sqrt = sqrt . abs
toFunc1 Square = (**2)

toFunc2 :: PrimitiveTwo -> (Float -> Float -> Float)
-- toFunc2 Div = (/)
toFunc2 Add = (+)
toFunc2 Sub = (-)
toFunc2 Mult = (*)


evaluate :: Node -> Float -> Float
evaluate (Value x) var = x
evaluate Var var = var
evaluate (DBranch func node1 node2) var = (toFunc2 func) (evaluate node1 var) (evaluate node2 var)
evaluate (SBranch func node1) var = (toFunc1 func) $ evaluate node1 var

data Genome = GenomeConstr {
                                res :: Float,
                                funcs :: [[Node]]
                                }
    deriving (Show, Read,Ord,Eq)
                                    
primitives1 = [cos, sin, sqrt . abs, (**2)]
primitives2 = [(+), (-), (*)]

myRound :: Float -> Float -> Float
myRound resol x = let pos =  if x >= 0 then head $ filter (\n -> (n >= -1.0 ) && (n <= 1.0)) $ iterate (\n -> n - 2.0) x
                                       else head $ filter (\n -> (n >= -1.0) && (n <= 1.0)) $ iterate ((+)2.0) x
                      r = resol
                  in if pos >= 0 then (fromIntegral $ floor $ pos * r) / r
                                 else (fromIntegral $ ceiling $ pos * r) / r



applyFuncs :: [Float] -> Genome -> [[Float]] 
applyFuncs vec genome = let fs = funcs genome
                            r = res genome
                            dims = length $ funcs genome
                        in concat [[update ind (myRound r $ (vec!!ind) + evaluate f (vec!!ind)) vec | f <- fs !! ind ] | ind <- [0..dims-1]]
                            
applyFuncsMult :: [[Float]] -> Genome -> [[Float]]                                
applyFuncsMult vecs genome= concatMap (\x -> applyFuncs x genome) vecs 
                                
locateSample :: Genome -> [Float] -> [[Float]]
locateSample genome sample = --just creates spacial vectors, no activations
    let factors = (28.0, 28.0)--head [(i, (length sample)/i) | i <- [floor . sqrt $ length sample .. 1], (length sample) `mod` i == 0]
        dimension = length $ funcs genome
        resIt = myRound $ res genome
        in let
            locPairs = [[resIt $ a/(fst factors) - 0.5 ,resIt $ b/(snd factors) - 0.5] | a <- [1..fst factors], b <- [1..snd factors]]
            empties = replicate (length sample) $ replicate (dimension - 2) 0
            in zipWith (++) locPairs empties


--CONSOLIDATION FUNCTIONS
--Consolidation List format:
--  A consolidation list should be a list of list of sublists, where each sublist contains the set
--  elements that overlap. These will be added to create a new element in their position.
--  if no other elements overlap an element, it will be a list of form [element].


consolGen :: Int -> [[Int]] -> [[Float]] -> [[Int]]
consolGen offset currList (v:vecs) = if (filter (elem offset) currList ) == [] 
    then consolGen (offset+1) (currList ++ [offset:( map (+(offset+1)) $ elemIndices v vecs)]) vecs
    else consolGen (offset + 1) currList vecs
consolGen offset currList [] = currList     

synapseNumThreshold :: Int
synapseNumThreshold = 250000
genConsolidate :: Genome -> [Float] -> [[[Int]]]
genConsolidate gene vecSample = 
    let inps = locateSample gene vecSample
        fs = funcs gene
        belowThreshold = \(outputVec, consols) -> ((length $ concat consols) + (length outputVec)*10) < synapseNumThreshold
        branching = length $ funcs gene
    in snd $ last $ filter belowThreshold $ take 2 $ iterate branch (inps, [])
        where branch (vec, cons) = let new = applyFuncsMult vec gene 
                                   in let consols = consolGen 0 [] new
                                      in (vecConsol new consols , cons ++ [consols])

vecConsol :: [[Float]] -> [[Int]] -> [[Float]]
vecConsol inps conslist = map (\l -> inps !! (head l) ) conslist --only deals with locations, so only gets unique locations


floatConsol :: [Float] -> [[Int]] -> [Float]
floatConsol inps conslist = map (\inds -> foldl' (+) 0 $ map (\x -> inps !! x) inds) conslist

-- consol :: Acc (Vector Float) -> [Acc (Vector Int)] -> Acc (Vector Float)   
-- consol inp conslist= concatMap (getAdd) conslist
--     where 
--         getAdd = fold (+) 0 $ A.map ((A.!!) inp) --fold is a accelerate function
        
        
--WEIGHT GENERATOR
genWeight :: (RandomGen g) => Int -> Rand g [Float]
genWeight n = sequence $ replicate n $ getRandomR (-1.0,1.0)

genWeights :: (RandomGen g) => [[[Int]]] -> Rand g [[Float]]
genWeights consols = sequence $ foldl' (\acc x -> acc ++ [genWeight $ length $ concat x] ) [] consols


--BIAS GENERATOR
genBiases :: (RandomGen g) => [[[Int]]] -> Rand g [[Float]]
genBiases consols = sequence $ foldl' (\acc x -> acc ++ [genBias $ length $ concat x]) [] consols
    where genBias n = sequence $ replicate n $ getRandomR (-2.0,2.0)

layer :: [Float] -> [[Int]] -> Int -> [Float] -> [Float] -> [Float]
layer input consol branching w b = 
    let branched = concat $ map (replicate branching) input 
    in map (tanh) . flip floatConsol consol . zipWith (+) b . zipWith (*) w $ branched

--FEEDFORWARD AND ERROR METHODS
feedForward :: [Float] -> [[[Int]]] -> Int -> [[Float]] -> [[Float]] -> [[Float]] -> [Float] -> [Float]
feedForward input consols branching ws bs fws fbs= 
    let initial = foldl' (\inp (w,b,cons)  -> layer inp cons branching w b ) input $ zip3 ws bs consols
    in map (tanh) . zipWith (+) fbs . map (\fw -> foldl' (+) 0.0 $ zipWith (*) initial fw) $ fws

predict :: [Float] -> [[[Int]]] -> Int -> [[Float]] -> [[Float]] -> [[Float]] -> [Float] -> Int
predict input consols branching ws bs fws fbs = 
    let fed = feedForward input consols branching ws bs fws fbs 
    in (\x -> x - 1) . head . (elemIndices (maximum fed)) $ fed


l2Error :: [Float] -> [Float] -> [[[Int]]] -> Int -> [[Float]] -> [[Float]] -> [[Float]] -> [Float] -> Float
l2Error input outp consols branching ws bs fws fbs = 
    let predicted = feedForward input consols branching ws bs fws fbs
    in sqrt $ foldl' (+) 0 $ map (**2) $ zipWith (-) outp predicted

intermediateFF input consols branching ws bs fws fbs=  
    let initial = foldl' (\(inp, inter) (w,b,cons)  -> let i = layer inp cons branching w b in (i, inter ++ [i])) (input,[]) $ zip3 ws bs consols
    in let out = map (tanh) . zipWith (+) fbs . map (\fw -> foldl' (+) 0.0 $ zipWith (*) (fst initial) fw) $ fws
        in (out, snd initial)
        
-- findIndexes :: [Int] -> [[Int]] -> [(Int,Int)]
-- findIndexes sublist singleLayerConsol = foldl' (\ind l -> (length $ elemIndices

updateMult :: [Int] -> [a] -> [a] -> [a]
updateMult inds replacemnts l = foldl' (\templist (replacemnt, ind) -> update ind replacemnt templist) l $ zip replacemnts inds

tanh' :: Float -> Float
tanh' x = 1 - (tanh x)**2

type DErrordInputs = [Float] --one per neuron, so one layer is [Float]
type LayerOutput = [Float]

gradLayer :: DErrordInputs -> LayerOutput -> Int -> [[Int]] -> [Float] -> [Float] -> ([Float], [Float], DErrordInputs)
gradLayer dEwrtInputs prevLayerOutput branching consols ws bs = (dErrordWeights, dErrordBiases, newdErrordInputs)
    where
        dErrordBiases = dEwrtInputs --dIn/dBias = 1
        
        expandedLayerOutput = concat $ map (replicate branching) prevLayerOutput
        expandedLayerOutputZeroed = replicate (length expandedLayerOutput) 0.0

        consolandInds = zip consols [0.. ((\x->x-1) $ length consols)]
        
        getSublistdEdW (consSublist, ind) = map (\consIndex -> (dEwrtInputs !! ind) * (expandedLayerOutput !! consIndex)) consSublist 
        dEdWsublists = map getSublistdEdW consolandInds
        --get list of dE/dw for each weight by multiplying dE/dIn * dIn/dOut = dE/dIn * w
        
        dErrordWeights = foldl' (\zeroList (dEdWsublist, consSublist) -> updateMult consSublist dEdWsublist zeroList) expandedLayerOutputZeroed $ zip dEdWsublists consols
        
        dOutsdIns = map (\x -> 1 - x**2) prevLayerOutput --remember dOut/dIn = 1 - (tanh in)**2 = 1- out**2
        --this finds the dE of each output neuron with respect to each input neuron by dE/dIn * dIn/dOut = dE/dIn * w
        --because each neuron connects to BRANCHING other neurons, each neuron has BRANCHING dE/dIn * dIn/dOut to be summed
        unsummeddEdOuts = foldl' (\weights (consSub, ind) -> updateMult consSub (map (\consInd -> (dEwrtInputs !! ind) * (ws !! consInd)) consSub) weights) ws consolandInds
        --sum up the BRANCHING dE/dIn*dIn/dOut   s for each neuron
        dEdOuts = map (sum) $ chunksOf branching unsummeddEdOuts
        dEdIns = DL.zipWith (*) dOutsdIns dEdOuts
        newdErrordInputs = dEdIns
        
         
        
    
    



grad :: [Float] -> [Float] -> [[[Int]]] -> Int -> [[Float]] -> [[Float]] -> [[Float]] -> [Float] -> [[[Float]]]
grad inp outp consols branching ws bs fws fbs = [dEdWs, dEdBs, dEdfws, [dEdfbs]]
    where 
        (final, layerOutputs) = intermediateFF inp consols branching ws bs fws fbs
        --error is the sum of half the squared difference between output and correct
        initdEdIns = map tanh' $ DL.zipWith (-) outp final
        finalStructuredLayerOut = last layerOutputs
        dEdfws = map (\dEdIn -> map (*dEdIn) finalStructuredLayerOut) initdEdIns
        dEdfbs = initdEdIns

        lastdEdIn = map (foldl' (+) 0.0) $ map (DL.zipWith (*) initdEdIns) $ transpose fws

        myGradLayer dEwrtInputs prevLayerOutput cons w b= gradLayer dEwrtInputs prevLayerOutput branching cons w b

        outputs = [inp] ++ layerOutputs
        zipped = zip4 consols ws bs outputs
        
        derivativeAppendtoList :: [[Float]] -> [[Float]] -> [Float] -> [[Int]] -> [Float] -> [Float] -> [Float] -> ([[Float]], [[Float]], [Float])
        derivativeAppendtoList gws gbs prevdEdIn cons w b out = let (gw, gb, newdEdIn)=myGradLayer prevdEdIn out cons w b
                                           in ([gw] ++ gws, [gb] ++ gbs, newdEdIn)

        (dEdWs,dEdBs,_) = foldr (\(cons, w, b, out) (gws,gbs, prevdEdIn)-> derivativeAppendtoList gws gbs prevdEdIn cons w b out) ([],[], lastdEdIn) zipped

         
         


alpha = 0.01
descend :: [[[Int]]] -> Int -> [Float] -> [Float] -> [[Float]] -> [[Float]] -> [[Float]] -> [Float] -> [[[Float]]]
descend consols branching inp outp ws bs fws fbs =
    let mygrads = grad inp outp consols branching ws bs fws fbs
    in  DL.zipWith (DL.zipWith (DL.zipWith (\g val -> val - g * alpha))) mygrads [ws,bs,fws,[fbs]] 

--GENETIC ALGORITHM

--
-- GA TYPE CLASS IMPLEMENTATION
--
maxFuncDepth = 4 :: Int
maxDimFuncs = 10
maxDims = 10
constantLims = (-5.0, 5.0) :: (Float,Float)
resolutionLimits = (30,100) :: (Int,Int)
resModifyRange = (0,5) :: (Int,Int)
epochs = 10
dimAddingRange = (0,5) :: (Int,Int)

selectRandom list g = (!! (fst $ randomR (0, (length list) - 1 ) g)) list

genRandomNode :: Int -> Int -> Node
genRandomNode 1 seed =  if (fst $ randomR (0,1) (mkStdGen seed) :: Int) == 0 then Value $ (fst $ randomR constantLims $ mkStdGen seed :: Float)
                                                                      else Var
genRandomNode n seed = let (nodetype, g)  = randomR (1,2) $ mkStdGen seed :: (Int, StdGen)
                       in let primitiveone = selectRandom [Cos, Sin, Sqrt, Square] g
                              primitivetwo = selectRandom [Add, Sub, Mult] g
                              s1 = fst $ random g
                              s2 = fst $ random $ mkStdGen s1
                          in if nodetype == 1 then SBranch (primitiveone) (genRandomNode (n-1) s1)
                                           else DBranch (primitivetwo) (genRandomNode (n-1) s1) (genRandomNode (n-1) s2)
genRandomNode _ seed = Var

genRandomDimNodes :: Int -> [Node]
--                                                  make new seed               add random node to list
genRandomDimNodes seed = snd $ foldl' (\(g,list) n -> (fst $ random $ mkStdGen g, list ++ [genRandomNode n g])) (seed, []) depths
    where depths = take (fst $ randomR (1, maxDimFuncs) $ mkStdGen seed) $ randomRs (1, maxFuncDepth) (mkStdGen seed)

duplicate l = map (\x -> [x,x]) l

myZipWith l1 l2 = let nl1 = length l1
                      nl2 = length l2
                  in if nl1 < nl2 then (take nl1 $ zipWith (\x y-> [x,y]) l1 l2) ++ (duplicate $ drop nl1 l2)
                                  else (take nl2 $ zipWith (\x y-> [x,y]) l1 l2) ++ (duplicate $ drop nl2 l2)

update index element list = take index list ++ [element] ++ drop (index + 1) list
dropAt n xs = let (ys,zs) = splitAt n xs   in   ys ++ (tail zs)

inc :: Int -> Int
inc x = x + 1


instance Entity Genome Float [[[Float]]] () IO where
  -- generate a random entity, i.e. a random string
  genRandom primitives seed = return $ GenomeConstr {res = fromIntegral resolution, funcs = dimLists}
    where
        g = mkStdGen seed
        (resolution, g1) = randomR resolutionLimits g
        (nDims,g2) = randomR (1,maxDims) g1
        seeds = randoms $ mkStdGen $ fst $ random g2
        
        dimLists = map (\s -> genRandomDimNodes s) $ take nDims seeds :: [[Node]]

  -- crossover operator: mix (and trim to shortest entity)
  crossover _ _ seed e1 e2 = return $ Just $ GenomeConstr {res = resol, funcs = e}
    where
      g = mkStdGen seed 
      cps = myZipWith (funcs e1) (funcs e2)
      picks = map (flip mod 2) $ randoms g
      e = zipWith (!!) cps picks
      resol = if (fst $ randomR (0,1) g :: Int) == 0 then res e1 else res e2

  mutation pool p seed e = return $ Just myFinal
    where
      g = mkStdGen seed
      numMutations = (round $ fromIntegral (length $ concat (funcs e)) * p) :: Int
      dimsToMutate = sort $ take numMutations $ randomRs (0, length (funcs e) -1) g  --sort might not be imported
      willAdd = (((fst $ randomR (0,1) g) :: Int) == 1)
      (aSeed,g1) = random g
      willAddDim = fst (randomR dimAddingRange g) == 0
      (dimToAddTo, g2) = randomR (0, length (funcs e) -1) g
      willAddorSub = take numMutations $ randomRs (0,20) g :: [Int]
      seeds = randoms g2 :: [Int]
      toModify = take (length dimsToMutate) $ zip3 dimsToMutate seeds willAddorSub
      
      toInd genSeed list = fst $ randomR (0, length list -1) $ mkStdGen genSeed
      genRand genSeed = genRandomNode (fst $ randomR (1,maxFuncDepth) $ mkStdGen genSeed) genSeed
      -- i=dim to mutate, s=random seed
      newNodes =  foldl'(\acc (i,s,a)-> update i ( if a /= 0 then update (toInd s (acc !! i)) (genRand s) (acc !! i) else dropAt (toInd s (acc !! i)) (acc !! i) ) acc) (funcs e) toModify  
      
      addMutated = if willAdd 
                     then if willAddDim
                         then newNodes ++ [genRandomDimNodes (fst $ random g1)]
                         else update dimToAddTo (newNodes !! dimToAddTo ++ [genRand aSeed]) newNodes
                     else if willAddDim
                         then update dimToAddTo [] newNodes
                         else let ind = toInd aSeed newNodes 
                              in update ind (update ( toInd (seeds !! 101) (newNodes !! ind) ) [] newNodes !! ind) newNodes
       
      myFinal = if ((fst $ randomR resModifyRange g1) == 1) 
                    then GenomeConstr {res = fromIntegral $ fst $ randomR resolutionLimits g1, funcs = addMutated}
                    else GenomeConstr {res = (res e), funcs = addMutated}



  -- NOTE: lower is better
  --dataset is [[trainimages], [trainlabels], [testimages], [testlabels]]
  score [trainImages, trainLabels, testImages, testLabels] genome = do
    let
        !cons =  genConsolidate genome $ head trainImages
        !branching = length . concat $ funcs genome
    
    !ws <- evalRandIO $ genWeights cons
    !bs <- evalRandIO $ genBiases cons
    !fws <- evalRandIO $ sequence $ replicate 10 $ genWeight $ length $ last cons --replicate 10 because 10 ouput neurons
    !fbs <- evalRandIO $ genWeight 10
    
    
    zipped <- return $ concat $ replicate (round $ epochs/10) $ zip trainImages trainLabels
    -- train:: [[[Float]]] -> [[[Float]]]
    let train [w,b,fw,[fb]] =  foldl' (\[tw,tb,tfw,[tfb]] (inp, outp)-> descend cons branching inp outp tw tb tfw tfb) [w,b,fw,[fb]] zipped
    weightlist <- return $ take 10 $ iterate train [ws, bs, fws, [fbs]]
    
    -- getTestAccuracy :: [[Float]] -> [[Float]] -> [[Float]] -> [Float] -> Int
    let getTestAccuracy w b fw fb = foldl' (\x (tI,tL)-> if (predict tI cons branching w b fw fb) == (head $ elemIndices 1.0 tL) then inc x else x) 0 $ zip testImages testLabels :: Int
    

    numCorrect <- return $ map (\[w,b,fw,[fb]] -> getTestAccuracy w b fw fb) weightlist
        
    
    errors <- return $ map (\x -> (10000.0 - (fromIntegral x))/100.0) (numCorrect :: [Int])
    fileNameNums <- evalRandIO $ getRandomR (1,10000000000) :: IO Int
    fileName <- return $ show $ fileNameNums
    writeFile fileName (show errors)
    return $ Just $ minimum errors
    

  -- whether or not a scored entity is perfect
  isPerfect (_,s) = s < 0.23


getImage s n = fromIntegral . BS.index s . (n*28^2 + 16 +) <$> [0..28^2 - 1]
getX     s n = (/ 256) <$> getImage s n
getLabel s n = fromIntegral $ BS.index s (n + 8)
getY     s n = fromIntegral . fromEnum . (getLabel s n ==) <$> [0..9]

main :: IO() 
main = do
        let cfg = GAConfig 
                    1 -- population size
                    50 -- archive size (best entities to keep track of)
                    1 -- maximum number of generations
                    0.8 -- crossover rate (% of entities by crossover)
                    0.2 -- mutation rate (% of entities by mutation)
                    0.0 -- parameter for crossover (not used here)
                    0.2 -- parameter for mutation (% of replaced letters)
                    False -- whether or not to use checkpointing
                    False -- don't rescore archive in each generation

            g = mkStdGen 0 -- random generator
        [trainIt, trainLt, testIt, testLt] <- mapM ((decompress  <$>) . BS.readFile) [ "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",  "t10k-images-idx3-ubyte.gz",  "t10k-labels-idx1-ubyte.gz"]
        
        trainI <- return $ map (getX trainIt) [1..60000] :: IO [[Float]]
        trainL <- return $ map (getY trainLt) [1..60000]
        testI <- return $ map (getX testIt) [1..10000]
        testL <- return $ map (getY testLt) [1..10000]
        
        
        -- Note: if either of the last two arguments is unused, just use () as a value
        es <- evolveVerbose g cfg () [trainI,trainL,testI,testL] :: IO (Archive Genome Float)
        let e = show $ snd $ head es :: String
        
        putStrLn $ "best entity (GA): " ++ (show e)
        writeFile "fractalResults" (show es)

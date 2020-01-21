 {-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE TypeSynonymInstances #-}
{-# LANGUAGE NamedFieldPuns #-}
{-# LANGUAGE BangPatterns #-}

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
import qualified Data.Time.Clock as T
import Data.Array.Accelerate as A
import Data.Array.Accelerate.CUDA as C

vdot :: VU.Vector Float -> VU.Vector Float -> Float
vdot x y = VU.foldl' (+) 0.0 $ VU.zipWith (*) x y

toIndexed2 dList =
	let
		indices = tail $ scanl (\acc next -> acc + (length next)) 0 dList
		concated = concat dList
	in (A.fromList (Z :. (length indices :: Int)) indices, A.fromList (Z :. (length concated :: Int)) concated)

mySlice a b v = A.take (b-a) $ A.drop a v
myLast (inds, l) = (A.length inds ==* 1) ? (mySlice 0 (A.head inds) l, mySlice (A.last $ A.init inds) (A.last inds) l)

relu = max 0

toTuple2 x y = (x,y)

type V3IndexList = Acc (A.Vector Int)
type V2IndexList = Acc (A.Vector Int) -- has no zero element at beginning. go from index corresponding to the first element of the pair to the index corresponding with the second element of the pair - 1
type V1Int = Acc (A.Vector Int)
type V1Float = Acc (A.Vector Float)
type V3IntIndexed = Acc (V3IndexList, V2IndexList, V1Int)
type V2FloatIndexed = Acc (V2IndexList, V1Float)

emptyVec = A.enumFromN (A.lift (Z :. (0 :: Int))) 0.0

gpuMap2 :: (V2IndexList, Acc (A.Vector a)) -> (Acc (A.Vector a) -> Acc (A.Vector e) ) -> (V2IndexList, Acc (A.Vector e))
gpuMap2 (inds, folding) f = iteration emptyVec emptyVec inds folding
    where
        iteration builtInds builtFold (indList, toFold) = if A.length indList > 1
            then
                let fd = f $ mySlice (indList !! 0) (indList !! 1 - 1) toFold
                in iteration (builtInds ++ (singleton $ (A.length fd) + (A.last builtInds)) (builtFold ++ Fd) (tail indList, drop (indList !! 1 - indList !! 0) toFold)
            else
                let fd = f $ toFold
                in (builtInds ++ (singleton $ (A.length fd) + (A.length builtFold) , builtFold ++ Fd)
                
                
gpuMap1to2 :: Acc (A.Vector a) -> (exp a -> Acc (A.Vector e)) -> (V2IndexList, Acc (A.Vector e))
gpuMap1to2 mapping f = toTuple2 $ iteration emptyVec emptyVec mapping
	where
		iteration builtInds builtMap toMap = if A.length toMap > 0:
			then
				let fd = f $ head toMap
				in iteration (builtInds ++ (singleton $ (A.length fd) + (last builtInds)) (builtMap ++ fd) (tail toMap)
			else
				builtInds builtMap

gpuScanL :: Acc (A.Vector b) -> (V2IndexList, Acc (A.Vector a)) -> (Acc (A.Vector b) -> Acc (A.Vector a) -> Acc (A.Vector e) ) -> (V2IndexList, Acc (A.Vector e))
gpuScanL initial (inds,scanning) f = iteration emptyVec emptyVec initial inds scanning
    where
        iteration builtInds builtScan lastOut (indList, toScan) = if A.length indList > 1
            then
                let fd = f lastOut $ mySlice (indList !! 0) (indList !! 1) toScan
                in iteration (builtInds ++ (singleton $ (A.length Fd) + (A.last builtInds))) (builtScan ++ Fd) Fd (tail indList, drop (indList !! 1 - indList !! 0) toScan
            else
                let fd = f lastOut $ toScan
                in (builInds ++ (singleton $ A.length fd) , builtScan ++ fd)
                
gpuScanLIndexed initial (inds, scanning) f = iteration emptyVec emptyVec (enumFromN (lift (Z :. (A.length inds))) 0) initial inds scanning
    where
        iteration builtInds builtScan iterIndex lastOut (indList, toScan) = if A.length indList > 1
            then
                let fd = f (A.head iterIndex) lastOut $ mySlice (indList !! 0) (indList !! 1) toScan
                in iteration (builtInds ++ (singleton $ A.length Fd)) (builtScan ++ Fd) (tail iterIndex) Fd (tail indList, drop (indList !! 1 - indList !! 0) toScan
            else
                let fd = f (A.head iterIndex) lastOut $ toScan
                in (builInds ++ (singleton $ A.length fd) , builtScan ++ fd)
    
                
-- gpuScanLPairs :: Acc (A.Vector a) -> (Acc (A.Vector a) -> Acc (A.Vector b)) -> (V2IndexList, Acc (A.Vector b))
-- gpuScanLPairs pairs f = iteration emptyVec emptyVec pairs
--     where
--         pairs = enum
--         iteration builtInds builtScan ps = if A.length pairs > 2
--             then
--                 let Fd = f $ A.take 2 pairs
--                 in iteration (builtInds ++ (singleton $ A.length Fd)) (builtScan ++ Fd) (tail pairs)
--             else
--                 let Fd = f pairs
--                 in (builtInds ++ (singleton $ A.length Fd), (builtScan ++ Fd))
 
-- gpuScanl12 :: Acc (Vector a) -> (Exp a -> Acc (A.Vector e) ) -> (V2IndexList, Acc (A.Vector e))
-- gpuScanl12  = iteration inds folding
--     where
--         iteration buildInds
-- type V1Float = VU.Vector Float
-- type V2Float = V.Vector ( VU.Vector Float)
-- type V3Int = V.Vector ( V.Vector (VU.Vector Int))
-- type V2Int = V.Vector (VU.Vector Int)
-- type V1Int = VU.Vector Int


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
primitives2 = [ (+), (-), (*)]

myRound :: Float -> Float -> Float
myRound !resol !x = let pos =  if x >= 0 then head $ filter (\n -> (n >= -1.0 ) && (n <= 1.0)) $ iterate (\n -> n - 2.0) x
                                       else head $ filter (\n -> (n >= -1.0) && (n <= 1.0)) $ iterate ((+)2.0) x
                        r = resol
                    in if pos >= 0 then (fromIntegral $ floor $ pos * r) / r
                                 else (fromIntegral $ ceiling $ pos * r) / r



applyFuncs :: [Float] -> Genome -> [[Float]]
applyFuncs !vec !genome = let fs = funcs genome
                              r = res genome
                              dims = length $ funcs genome
                          in concat [[update ind (myRound r $ (vec!!ind) + evaluate f (vec!!ind)) vec | f <- fs !! ind ] | ind <- [0..dims-1]]
                            
applyFuncsMult :: [[Float]] -> Genome -> [[Float]]
applyFuncsMult !vecs !genome= concatMap (\x -> applyFuncs x genome) vecs
                                
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
consolGen !offset !currList !(v:vecs) = if (filter (elem offset) currList ) == []
    then consolGen (offset+1) (currList ++ [offset:( map (+(offset+1)) $ elemIndices v vecs)]) vecs
    else consolGen (offset + 1) currList vecs
consolGen !offset !currList [] = currList

synapseNumThreshold :: Int
synapseNumThreshold = 200000
genConsolidate :: Genome -> Int -> [Float] -> [[[Int]]]
genConsolidate !gene !outsize !vecSample =
    let inps = locateSample gene vecSample
        fs = funcs gene
        belowThreshold = \(outputVec, consols) -> ((length $ concat $ concat consols) + (length outputVec)*outsize) < synapseNumThreshold
        branching = length $ funcs gene
    in snd $ last $ takeWhile belowThreshold $ iterate branch (inps, [])
        where branch (!vec, !cons) = let new = applyFuncsMult vec gene
                                   in let consols = consolGen 0 [] new
                                      in (vecConsol new consols , cons ++ [consols])

vecConsol :: [[Float]] -> [[Int]] -> [[Float]]
vecConsol inps conslist = map (\l -> inps !! (head l) ) conslist --only deals with locations, so only gets unique locations


floatConsol :: [Float] -> [[Int]] -> [Float]
floatConsol inps conslist = map (\inds -> foldl' (+) 0 $ map (\x -> inps !! x) inds) conslist

floatConsolVec :: V1Float -> V2Int -> V1Float
floatConsolVec inps conslist= G.convert $ V.map (\inds -> VU.foldl' (\acc ind-> acc + inps VU.! ind) 0.0 inds) conslist


gpuFloatConsol :: V2IndexList -> V1Int -> V1Float -> V1Float
gpuFloatConsol indList cons inp = stencil (myAdd) 0 indList
    where
        myAdd :: A.Stencil3 -> Exp Float
        myAdd (a , b , c) = A.fold (+) 0.0 $ mySlice a b inp
    
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
genBiases consols = sequence $ foldl' (\acc x -> acc ++ [genBias $ length x]) [] consols
    where genBias n = sequence $ replicate n $ getRandomR (-2.0,2.0)
    
    
--FEEDFORWARD AND ERROR METHOD
    
    

gpuLayer :: V1Float -> V2IntIndexed -> Int -> V1FLoat -> V1Float -> V1Float
gpuLayer inp (consIndex, cons) branching w b = A.map (max 0) . A.zipWith (+) b $ gpuFloatConsol consIndex cons $ A.zipWith (*) w branched
    where branched = A.flatten $ A.replicate (A.lift (Z :. All :. branching)) inp

gpuIntermediateFF :: V1Float -> V3IntIndex -> Acc (Scalar Int) -> V2FloatIndex -> V2FloatIndexed -> Acc (A.Array DIM2 Float) -> V1Float -> (V1Float, V2FloatIndexed)
gpuIntermediateFF inp (consIndexIndex, consIndex, cons) branching (wIndex, ws) (bIndex, bs) fws fbs =
    (A.map (max 0) $ A.zipWith (+) fbs $ multiplied, (interIndex, intermediates))
    where
        myLayer = \i conIndexs lastOut -> gpuLayer lastOut (consIndexs, cons) branching (ws !! i) (bs !! i)
        (interIndex, intermediates) = gpuScanLIndexed myLayer consIndexIndex consIndex
        lastOut =  mySlice (interIndex !! (A.length interIndex - 2)) (interIndex !! (A.length interIndex -1)) intermediates
        multiplied = A.fold (+) 0 $ zipWith (*) (A.replicate (A.lift (Z :. All :. 10)) lastOut) fws
       
-- POSSIBLE BUG SOURCE, THE +1 HERE
                    -- Might need a lift in the enumFromN
       
       
-- gpufeedForward :: V1Float -> V3IndexList -> V2IndexList -> V1Int -> Int -> V2IndexList -> V1Float -> V2IndexList -> V1Float -> V2IndexList -> V1Float -> V1Float -> V1Float
-- gpufeedForward input consIndex3List consIndex2List consList branching wIndexList ws bIndexList bs fwIndexList fws fbs=
--     VU.map (tanh) . VU.zipWith (+) fbs . G.convert . V.map (\fw -> vdot initial fw) $ fws
--         where initial = A.fold (\inp (w,b,cons)  -> layer inp cons branching w b ) input $ V.zip3 ws bs consols
        


--unboxed
predict :: V1Float -> V3Int -> Int -> V2Float -> V2Float -> V2Float -> V1Float -> Int
predict input consols branching ws bs fws fbs =
    VU.head $ VU.elemIndices (VU.maximum fed) fed
        where fed = fst $ intermediateFF inputs consols branching ws bs fws fbs



tanh' :: Float -> Float
tanh' x = 1 - (tanh x)**2

type DErrordInputs = V1Float --one per neuron, so one layer is [Float]
type LayerOutput = V1Float

vLast vec = vec !! (A.length vec -1)

relu' x = (x >* (A.constant 0.0)) ? ((A.constant 1.0), (A.constant 0.0))

--unboxed
gpuGradLayer :: DErrordInputs -> LayerOutput -> Acc (Scalar Int) -> V2IntIndex -> V1Float -> V1Float -> (V1Float, V1Float, DErrordInputs)
gpuGradLayer dEwrtInputs prevLayerOutput branching (consInds, consols) ws bs = (dErrordWeights, dEwrtInputs, newdErrordInputs)
    where
        expandedLayerOutput = A.flatten $ A.replicate (A.lift (Z :. All :. branching)) prevLayerOutput
        expandedLayerOutputZeroed = generate (index1 $ A.length expandedLayerOutput) (\_ -> 0.0)
        
        myCons = mySlice (A.head consInds) (vLast consInds -1) consols
        dEdWs = A.map (\ind -> (dEwrtInputs !! ind) * (expandedLayerOutput !! ind)) myCons
        dErrordWeights = A.scatter myCons expandedLayerOutputZeroed dEdWs
        
        dOutsdIns = A.map (\x -> if x > 0 then 1 else 0) prevLayerOutput
        inds = enumFromN 0 (A.length dOutsdIns - 2) -- possible bug with the 2
        
        dEdOutsunplaced = gpuMap1to2 (\ind -> A.map (\consInd -> (dEwrtInputs A.!! ind) * (ws A.!! consInd)) $ mySlice (consInd A.!! ind) (consInd A.!! (ind + 1)) consols) inds
        
        unsummeddEdOuts = A.scatter myCons expandedLayerOutputZeroed dEdOutsunplaced
        
        dEdOuts = A.fold (+) 0 $ A.transpose $ A.reshape (A.lift (Z :. branching :. All)) unsummeddEdOuts
        --possible bug in reshape and whether the rows are actually the groups we want? Maybe don't need transpose?
        newdErrordInputs = A.zipwith (*) dEdOuts dOutsdIns
        
      
       
gpuGrad :: V1Float -> V1Float -> V3IntIndexed -> Acc (Scalar Int) -> V2FloatIndexed -> V2FloatIndexed -> V2FloatIndexed -> V2FloatIndexed -> Acc (A.Vector Float, A.Vector Float, A.Vector Float,  A.Vector Float)
gpuGrad inp outp consols branching (wInds,ws) (bInds,bs) (fwInds,fws) fbs = (dEdWs, dEdBs, dEdfws, initdEdIns)
	where
		(final, (layerOutputsInds, layerOutputs)) = gpuIntermediateFF inp consols branching ws bs fws fbs
		finaldOutdIn = A.map (\x -> (x >* (constant 0.0)) ? (x, constant 0.0) final
		
		initdEdIns = A.zipwith (*) finaldOutdIn $ A.zipWith (-) outp final
		
		finalStructuredLayerOut = mySlice (A.last $ A.init layerOutputInds) (A.last layerOutputInds) layerOutputs
		
		(dEdfwsInds, dEdfws) = gpuMap1to2 (\dEdIn -> A.map (*dEdIn) finalStructuredLayerOut) initdEdIns
		
		finalStructuredLayerdOutdIn = A.map relu' finalStructuredLayerOut
		
		finalStrucLength = A.length finalStructuredLayerOut
		numOutputs = A.length fwInds
		numOutEnum = A.enumFrom (A.lift (Z :. (numOutputs))) (A.constant (0.0 :: Float))
		numOutputInsEnum = A.enumFrom (A.lift (Z :. finalStrucLength)) (A.constant (0.0 :: Float))
		
		finalStructuredLayerdEdOut = A.map (\i -> A.fold (+) 0 $ zipWith (*) initdEdIns $
																				A.map (\x -> fws !! (i + finalStrucLength * x)) numOutEnum) numOutputInsEnum
		
		finalStructuredLayerdEdIn = A.zipWith (*) finalStructuredLayerdEdOut finalStructureLayerdOutdIn
		
		
		
		derivativeAppend wInds bInds consIndInds (outInds,prevOuts) newdEdIn = if A.length wInds ==* 0 then A.lift (empytVec, emptyVec) else
			let prevOut = myLast outInds prevOuts
					w = myLast ws
					b = myLast bs
					consInds = myLast consIndInds
			in let
				(dEdWs, dEdBs, nextdEdIn) = A.unlift $ gpuGradLayer newdEdIn prevOut branching (consInds, consols) w b
				 in A.lift (laterdEdWs ++ dEdWs, laterdEdBs ++ dEdBs)
				 		where (laterdEdWs, laterdEdBs) = A.unlift $ derivativeAppend (A.init wInds) (A.init bInds) (A.init consIndInds
        

         
        
toTuple [x,y,z,a] = (x,y,z,a)
toList4 (x,y,z,a) = [x,y,z,a]

alpha = 0.005
descend :: V3Int -> Int -> V1Float -> V1Float -> V2Float -> V2Float -> V2Float -> V2Float -> (V2Float, V2Float,V2Float,V2Float)
descend consols branching inp outp ws bs fws fbsI =
    let mygrads = grad inp outp consols branching ws bs fws (V.head fbsI)
    in  toTuple $ DL.zipWith (V.zipWith (VU.zipWith (\g val -> val - g * alpha))) mygrads [ws,bs,fws,fbsI]

--GENETIC ALGORITHM

--
-- GA TYPE CLASS IMPLEMENTATION
--
maxFuncDepth = 4 :: Int
maxDimFuncs = 8
maxDims = 5
constantLims = (-5.0, 5.0) :: (Float,Float)
resolutionLimits = (30,50) :: (Int,Int)
resModifyRange = (0,5) :: (Int,Int)
epochs = 5
dimAddingRange = (0,5) :: (Int,Int)

selectRandom list g = (!! (fst $ randomR (0, (length list) - 1) g)) list

genRandomNode :: Int -> Int -> Node
genRandomNode 0 seed = Var
genRandomNode 1 seed =  if (fst $ randomR (0,1) (mkStdGen seed) :: Int) == 0 then Value $ (fst $ randomR constantLims $ mkStdGen seed :: Float)
                                                                      else Var
genRandomNode n seed = let (nodetype, g)  = randomR (1,2) $ mkStdGen seed :: (Int, StdGen)
                       in let primitiveone = selectRandom [Cos, Sin, Sqrt, Square] g
                              primitivetwo = selectRandom [Add, Sub, Mult] g
                              s1 = fst $ random g
                              s2 = fst $ random $ mkStdGen s1
                          in if nodetype == 1 then SBranch (primitiveone) (genRandomNode (n-1) s1)
                                           else DBranch (primitivetwo) (genRandomNode (n-1) s1) (genRandomNode (n-1) s2)

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


-- train cons brancing imgs labels [w,b,fw,fb] =  foldl' (\(!tw,!tb,!tfw,!tfb) (!inp,!outp) -> descend cons branching (V.fromList inp) (V.fromList outp) tw tb tfw tfb) (w,b,fw,fb) $ zip imgs labels -- zipped


instance Entity Genome Float (V2Float,V2Float,V2Float,V2Float) () IO where
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
  score (trainImages, trainLabels, testImages, testLabels) !genome = do
    putStrLn $ "scoring individual with res " ++ (show $ res genome)
    let
        !consols = V.fromList . map V.fromList . map (map VU.fromList) $ genConsolidate genome 10 [1.0..784.0]
        branching = length . concat $ funcs genome
    let consList = V.toList $ V.map V.toList $ V.map (V.map (VU.toList)) $ consols :: [[[Int]]]
    putStrLn "\nconsolidation list generated...\n"
    putStrLn $ (show $ length consList) ++ " layers"
    putStrLn $ show genome
    
    putStrLn $ "synapse number: " ++ ( show ((length $ last consList) * 10 + (length $ concat $ concat consList)))
    
    
    wsTemp <- evalRandIO $ genWeights $ consList :: IO [[Float]]
    !ws <- return $ toIndexed2 wsTemp :: IO (V2FloatIndexed)
    bsTemp <- evalRandIO  $ genBiases consList :: IO [[Float]]
    !bs <- return $ toIndexed2 bsTemp ::  IO (V2FloatIndexed)
    fwsTemp <- evalRandIO . Prelude.sequence $ Prelude.replicate 10 $ genWeight (length $ last consList) :: IO [[Float]] --replicate10 b/c 10 out neurons
    !fws <- return $ toIndexed2 fwsTemp :: IO (V2FloatIndexed)
    fbsTemp <- evalRandIO $ Prelude.sequence $ Prelude.replicate 1 $ genWeight 10 :: IO [[Float]]
    !fbs <- return $ A.fromList (Z :. (1 :: Int) :. (length $ head fbsTemp)) fbsTemp :: IO (Array DIM2 Float)
    
    
    putStrLn $ "descending once"
    startTime <- T.getCurrentTime
    putStrLn $ show $ map (V.map (VU.length)) $ toList4 $ descend consols branching ( V.head trainImages) (V.head trainLabels) ws bs fws fbs
    finishTime <- T.getCurrentTime
    putStrLn $ "finished descent" ++ (show $ T.diffUTCTime finishTime startTime)
    
    putStrLn $ "\nstarting to score genome: " ++ (show genome)
    startTime <- T.getCurrentTime
    -- zipped <- return $ concat $ replicate (round $ epochs/10) $ zip trainImages (trainLabels :: [V1Float])
    -- train:: [[[Float]]] -> [[[Float]]]
    let train [w,b,fw,fb] =  V.foldl' (\(!tw,!tb,!tfw,!tfb) (inp,outp) -> descend consols branching inp outp tw tb tfw tfb) (w,b,fw,fb) $ V.zip trainImages trainLabels -- zipped
    
    !weightlist <- return $ scanl' (\l _ -> toList4 $ train l) [ws,bs,fws,fbs] $ replicate epochs 0

    -- weightlist <- let wlist = tail $ take 2 $ iterate train [ws, bs, fws, fbs] in return wlist :: IO [[V2Float]]
    
    putStrLn $ "-Debug " ++ (show $ map (map (V.map VU.sum)) weightlist)
    endTime <- T.getCurrentTime
    putStrLn $ show $ T.diffUTCTime endTime startTime
    putStrLn "Testing Genome..."
    
    let getTestAccuracy w b fw fb = V.foldl' (\acc (tI,tL)-> if (predict tI consols branching w b fw fb) == (VU.head $ VU.elemIndices 1.0 tL) then acc + 1 else acc ) 0 $ V.zip testImages testLabels :: Int
    
    numCorrect <- return $ map (\[w,b,fw,fb] -> getTestAccuracy w b fw (V.head fb)) weightlist
        
    
    errors <- return $ map (\x -> (10000.0 - (fromIntegral x))/100.0) numCorrect
    fileNameNums <- evalRandIO $ getRandomR (1,10000000000) :: IO Int
    fileName <- return $ show $ fileNameNums
    writeFile fileName (show errors)
    putStrLn $ "-SCORELIST- " ++ (show errors)
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
                    20 -- population size
                    50 -- archive size (best entities to keep track of)
                    20-- maximum number of generations
                    0.8 -- crossover rate (% of entities by crossover)
                    0.2 -- mutation rate (% of entities by mutation)
                    0.0 -- parameter for crossover (not used here)
                    0.2 -- parameter for mutation (% of replaced letters)
                    False -- whether or not to use checkpointing
                    False -- don't rescore archive in each generation

            g = mkStdGen 0 -- random generator
        putStrLn "starting..."
        [trainIt, trainLt, testIt, testLt] <- mapM ((decompress  <$>) . BS.readFile) [ "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",  "t10k-images-idx3-ubyte.gz",  "t10k-labels-idx1-ubyte.gz"]
        
        putStrLn "decompressed images and labels \n"
        
        trainIt <- return $ map (getX trainIt) [0..49999] :: IO [[Float]]
        trainI <- return $ V.fromList $ map (VU.fromList) $ trainIt
        -- unique <- return $ length $ filter (/=784) $ map (length) trainI
        -- putStrLn $ "Number of training images with length /=784  " ++ (show $ unique)
        trainLt <- return  $ map (getY trainLt) [0..49999] :: IO [[Float]]
        trainL <- return $ V.fromList $ map (VU.fromList) $ trainLt
        testIt <- return $ map (getX testIt) [0..9999] :: IO [[Float]]
        testI <- return $ V.fromList $ map (VU.fromList) $ testIt
        testLt <- return $ map (getY testLt) [0..9999] :: IO [[Float]]
        testL <- return $ V.fromList $ map (VU.fromList) $ testLt
        
        putStrLn "Converted. Let us begin...\n"
        
        -- Note: if either of the last two arguments is unused, just use () as a value
        es <- evolveVerbose g cfg () (trainI,trainL,testI,testL) :: IO (Archive Genome Float)
        let e = show $ snd $ head es :: String
        
        putStrLn $ "best entity (GA): " ++ (show e)
        writeFile "fractalResults" (show es)

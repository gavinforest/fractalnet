import Codec.Compression.GZip (decompress)
import qualified Data.ByteString.Lazy as BS
import Control.Monad
import Data.Functor
import Data.Ord



getImage s n = fromIntegral . BS.index s . (n*28^2 + 16 +) <$> [0..28^2 - 1]
getX     s n = (/ 256) <$> getImage s n
getLabel s n = fromIntegral $ BS.index s (n + 8)
getY     s n = fromIntegral . fromEnum . (getLabel s n ==) <$> [0..9]


main = do
	[trainIt, trainLt, testIt, testLt] <- mapM ((decompress  <$>) . BS.readFile) [ "train-images-idx3-ubyte.gz","train-labels-idx1-ubyte.gz", "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]
        
        trainI <-return $ map (getX trainIt) [1..50000]
        trainL <- return $ map (getY trainLt) [1..50000]
        testI <- return $ map (getX testIt) [1..10000]
        testL <- return $ map (getY testLt) [1..10000]
        dataset <- return [trainI, trainL, testI, testL]
	putStrLn $ show $ dataset !! 0 !! 49999
	putStrLn $ show $ dataset !! 1 !! 0
	putStrLn $ show $ dataset !! 2 !! 0
	putStrLn $ show $ dataset !! 3 !! 0

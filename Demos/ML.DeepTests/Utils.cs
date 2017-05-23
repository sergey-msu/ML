using System;
using System.IO;
using System.Linq;
using ML.Contracts;
using ML.Core;
using ML.DeepMethods.Algorithms;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;

namespace ML.DeepTests
{
  public static class Utils
  {
    public static void HandleClassificationEpochEnded(BackpropAlgorithm alg,
                                                      MultiRegressionSample<double[][,]> test,
                                                      MultiRegressionSample<double[][,]> train,
                                                      Class[] classes,
                                                      string outputPath)
    {
      Console.WriteLine("---------------- Epoch #{0} ({1})", alg.Epoch, DateTime.Now);
      Console.WriteLine("L:\t{0}", alg.LossValue);
      Console.WriteLine("DW:\t{0}", alg.Step2);
      Console.WriteLine("LR:\t{0}", alg.LearningRate);

      double? pct = null;

      if (test==null || !test.Any())
        Console.WriteLine("Test: none");
      else
      {
        var terrors = alg.GetClassificationErrors(test, classes);
        var tec = terrors.Count();
        var tdc = test.Count;
        var tpct = Math.Round(100.0F * tec / tdc, 2);
        Console.WriteLine("Test: {0} of {1} ({2}%)", tec, tdc, tpct);

        pct = tpct;
      }

      if (train==null || !train.Any())
        Console.WriteLine("Train: none");
      else
      {
        var verrors = alg.GetClassificationErrors(train, classes);
        var vec = verrors.Count();
        var vdc = train.Count;
        var vpct = Math.Round(100.0F * vec / vdc, 2);
        Console.WriteLine("Train: {0} of {1} ({2}%)", vec, vdc, vpct);

        if (!pct.HasValue) pct=vpct;
      }

      var ofileName = string.Format("cn_e{0}_p{1}.mld", alg.Epoch, Math.Round(pct.Value, 2));
      var ofilePath = Path.Combine(outputPath, ofileName);
      using (var stream = File.Open(ofilePath, FileMode.Create))
      {
        alg.Net.Serialize(stream);
      }
    }

    public static void HandleRegressionEpochEnded(BackpropAlgorithm alg,
                                                  MultiRegressionSample<double[][,]> test,
                                                  MultiRegressionSample<double[][,]> train,
                                                  string outputPath)
    {
      Console.WriteLine("---------------- Epoch #{0} ({1})", alg.Epoch, DateTime.Now);
      Console.WriteLine("L:\t{0}", alg.LossValue);
      Console.WriteLine("DW:\t{0}", alg.Step2);
      Console.WriteLine("LR:\t{0}", alg.LearningRate);

      var rerror = 0.0D;
      if (test==null || !test.Any())
        Console.WriteLine("Test: none");
      else
      {
        rerror = alg.GetRegressionError(test);
        Console.WriteLine("Test error: {0}", Math.Round(rerror, 2));
      }

      if (train==null || !train.Any())
        Console.WriteLine("Train: none");
      else
      {
        var terror = alg.GetRegressionError(train);
        Console.WriteLine("Train error: {0}", Math.Round(terror, 2));
      }

      var ofileName = string.Format("cn_e{0}_regerr_{1}.mld", alg.Epoch, Math.Round(rerror, 2));
      var ofilePath = Path.Combine(outputPath, ofileName);
      using (var stream = File.Open(ofilePath, FileMode.Create))
      {
        alg.Net.Serialize(stream);
      }
    }

    public static void HandleBatchEnded(BackpropAlgorithm alg, int trainCount, DateTime tstart)
    {
      var now = DateTime.Now;
      var iter = alg.Iteration;
      var pct = Math.Min(100*iter/(float)trainCount, 100);
      var elapsed = TimeSpan.FromMinutes((now-tstart).TotalMinutes * (trainCount-iter)/iter);
      Console.Write("\rCurrent epoch progress: {0:0.00}%. Left {1:00}m {2:00}s.  L={3:0.0000}         ",
                    pct,
                    elapsed.Minutes,
                    elapsed.Seconds,
                    alg.LossValue);
    }

    public static void SaveAlgCrushResults(BackpropAlgorithm alg, string outputPath)
    {
      var ofileName = string.Format("cn_e{0}_crush.mld", alg.Epoch);
      var ofilePath = Path.Combine(outputPath, ofileName);
      using (var stream = File.Open(ofilePath, FileMode.Create))
      {
        alg.Net.Serialize(stream);
      }
    }


    public static void Convolve(string srcPath, string kernelPath, string outPath)
    {
      var data = new double[1][,] { new double[32,32] };
      var ker = new double[1][,] { new double[9,9] };

      using (var cat    = new System.Drawing.Bitmap(srcPath))
      using (var kernel = new System.Drawing.Bitmap(kernelPath))
      {
        for (int x=0; x<32; x++)
        for (int y=0; y<32; y++)
          data[0][y, x] = (255.0D-cat.GetPixel(x, y).B)/255.0D;

        for (int x=0; x<9; x++)
        for (int y=0; y<9; y++)
          ker[0][y, x] = (255.0D-kernel.GetPixel(x, y).B)/255.0D;
      }

      var convolution = new DeepMethods.Models.ConvLayer(1, 9, padding: 4);
      convolution.InputDepth = 1;
      convolution.InputHeight = 32;
      convolution.InputWidth = 32;
      convolution._Build();

      for (int y=0; y<9; y++)
      for (int x=0; x<9; x++)
        convolution.SetKernel(0, 0, y, x, ker[0][y, x]);

      var res = convolution.Calculate(data);
      var max = res[0].Cast<double>().Max();

      using (var fm = new System.Drawing.Bitmap(32, 32))
      {
        for (int x=0; x<32; x++)
        for (int y=0; y<32; y++)
        {
          var val = 255.0D*(1.0D - res[0][y, x]/max);
          if (val>128) val = 255;
          fm.SetPixel(x, y, System.Drawing.Color.FromArgb((int)val, (int)val, (int)val));
        }

        fm.Save(outPath);
      }
    }

    public static void ExportImageData(double[][,] data, string fpath)
    {
      var path = Path.GetDirectoryName(fpath);
      if (!Directory.Exists(path)) Directory.CreateDirectory(path);

      var height = data[0].GetLength(0);
      var width = data[0].GetLength(1);

      using (var ofile = File.Open(fpath, FileMode.Create, FileAccess.Write))
      {
        var image = new System.Drawing.Bitmap(width, height);

        for (int y=0; y<height; y++)
        for (int x=0; x<width; x++)
        {
          var rmap = data[0];
          var gmap = (data.Length>1) ? data[1] : data[0];
          var bmap = (data.Length>2) ? data[2] : data[0];

          var r = (int)(rmap[y, x]*255);
          var g = (int)(gmap[y, x]*255);
          var b = (int)(bmap[y, x]*255);
          image.SetPixel(x, y, System.Drawing.Color.FromArgb(r, g, b));
        }

        image.Save(ofile, System.Drawing.Imaging.ImageFormat.Png);

        ofile.Flush();
      }
    }

    public static void Shuffle<TSample>(ref TSample sample)
      where TSample : MarkedSample<double[][,], double[]>, new()
    {
      var result = new TSample();

      var cnt = sample.Count;
      var ids = Enumerable.Range(0, cnt).ToList();
      var random = new Random(0);

      var res = cnt;
      for (int i=0; i<cnt; i++)
      {
        var pos = random.Next(res--);
        var idx = ids[pos];
        ids.RemoveAt(pos);

        var data = sample.ElementAt(idx);
        result[data.Key] = data.Value;
      }

      sample = result;
    }

    #region Images

    public static class Matrix
    {
        public static double[,] Laplacian3x3
        {
            get
            {
                return new double[,]
                { { -1, -1, -1,  },
                  { -1,  8, -1,  },
                  { -1, -1, -1,  }, };
            }
        }

        public static double[,] Laplacian5x5
        {
            get
            {
                return new double[,]
                { { -1, -1, -1, -1, -1, },
                  { -1, -1, -1, -1, -1, },
                  { -1, -1, 24, -1, -1, },
                  { -1, -1, -1, -1, -1, },
                  { -1, -1, -1, -1, -1  }, };
            }
        }

        public static double[,] LaplacianOfGaussian
        {
            get
            {
                return new double[,]
                { {  0,   0, -1,  0,  0 },
                  {  0,  -1, -2, -1,  0 },
                  { -1,  -2, 16, -2, -1 },
                  {  0,  -1, -2, -1,  0 },
                  {  0,   0, -1,  0,  0 }, };
            }
        }

        public static double[,] Gaussian3x3
        {
            get
            {
                return new double[,]
                { { 1, 2, 1, },
                  { 2, 4, 2, },
                  { 1, 2, 1, }, };
            }
        }

        public static double[,] Gaussian5x5Type1
        {
            get
            {
                return new double[,]
                { { 2, 04, 05, 04, 2 },
                  { 4, 09, 12, 09, 4 },
                  { 5, 12, 15, 12, 5 },
                  { 4, 09, 12, 09, 4 },
                  { 2, 04, 05, 04, 2 }, };
            }
        }

        public static double[,] Gaussian5x5Type2
        {
            get
            {
                return new double[,]
                { {  1,   4,  6,  4,  1 },
                  {  4,  16, 24, 16,  4 },
                  {  6,  24, 36, 24,  6 },
                  {  4,  16, 24, 16,  4 },
                  {  1,   4,  6,  4,  1 }, };
            }
        }

        public static double[,] Sobel3x3Horizontal
        {
            get
            {
                return new double[,]
                { { -1,  0,  1, },
                  { -2,  0,  2, },
                  { -1,  0,  1, }, };
            }
        }

        public static double[,] Sobel3x3Vertical
        {
            get
            {
                return new double[,]
                { {  1,  2,  1, },
                  {  0,  0,  0, },
                  { -1, -2, -1, }, };
            }
        }

        public static double[,] Prewitt3x3Horizontal
        {
            get
            {
                return new double[,]
                { { -1,  0,  1, },
                  { -1,  0,  1, },
                  { -1,  0,  1, }, };
            }
        }

        public static double[,] Prewitt3x3Vertical
        {
            get
            {
                return new double[,]
                { {  1,  1,  1, },
                  {  0,  0,  0, },
                  { -1, -1, -1, }, };
            }
        }


        public static double[,] Kirsch3x3Horizontal
        {
            get
            {
                return new double[,]
                { {  5,  5,  5, },
                  { -3,  0, -3, },
                  { -3, -3, -3, }, };
            }
        }

        public static double[,] Kirsch3x3Vertical
        {
            get
            {
                return new double[,]
                { {  5, -3, -3, },
                  {  5,  0, -3, },
                  {  5, -3, -3, }, };
            }
        }
    }

    public static class Filters
    {
        private static Bitmap ConvolutionFilter(Bitmap sourceBitmap,
                                             double[,] filterMatrix,
                                                  double factor = 1,
                                                       int bias = 0,
                                             bool grayscale = false)
        {
            BitmapData sourceData = sourceBitmap.LockBits(new Rectangle(0, 0,
                                     sourceBitmap.Width, sourceBitmap.Height),
                                                       ImageLockMode.ReadOnly,
                                                 PixelFormat.Format32bppArgb);

            byte[] pixelBuffer = new byte[sourceData.Stride * sourceData.Height];
            byte[] resultBuffer = new byte[sourceData.Stride * sourceData.Height];

            Marshal.Copy(sourceData.Scan0, pixelBuffer, 0, pixelBuffer.Length);
            sourceBitmap.UnlockBits(sourceData);

            if (grayscale == true)
            {
                float rgb = 0;

                for (int k = 0; k < pixelBuffer.Length; k += 4)
                {
                    rgb = pixelBuffer[k] * 0.11f;
                    rgb += pixelBuffer[k + 1] * 0.59f;
                    rgb += pixelBuffer[k + 2] * 0.3f;


                    pixelBuffer[k] = (byte)rgb;
                    pixelBuffer[k + 1] = pixelBuffer[k];
                    pixelBuffer[k + 2] = pixelBuffer[k];
                    pixelBuffer[k + 3] = 255;
                }
            }

            double blue = 0.0;
            double green = 0.0;
            double red = 0.0;

            int filterWidth = filterMatrix.GetLength(1);
            int filterHeight = filterMatrix.GetLength(0);

            int filterOffset = (filterWidth-1) / 2;
            int calcOffset = 0;

            int byteOffset = 0;

            for (int offsetY = filterOffset; offsetY <
                sourceBitmap.Height - filterOffset; offsetY++)
            {
                for (int offsetX = filterOffset; offsetX <
                    sourceBitmap.Width - filterOffset; offsetX++)
                {
                    blue = 0;
                    green = 0;
                    red = 0;

                    byteOffset = offsetY *
                                 sourceData.Stride +
                                 offsetX * 4;

                    for (int filterY = -filterOffset;
                        filterY <= filterOffset; filterY++)
                    {
                        for (int filterX = -filterOffset;
                            filterX <= filterOffset; filterX++)
                        {

                            calcOffset = byteOffset +
                                         (filterX * 4) +
                                         (filterY * sourceData.Stride);

                            blue += (double)(pixelBuffer[calcOffset]) *
                                    filterMatrix[filterY + filterOffset,
                                                        filterX + filterOffset];

                            green += (double)(pixelBuffer[calcOffset + 1]) *
                                     filterMatrix[filterY + filterOffset,
                                                        filterX + filterOffset];

                            red += (double)(pixelBuffer[calcOffset + 2]) *
                                   filterMatrix[filterY + filterOffset,
                                                      filterX + filterOffset];
                        }
                    }

                    blue = factor * blue + bias;
                    green = factor * green + bias;
                    red = factor * red + bias;

                    if (blue > 255)
                    { blue = 255; }
                    else if (blue < 0)
                    { blue = 0; }

                    if (green > 255)
                    { green = 255; }
                    else if (green < 0)
                    { green = 0; }

                    if (red > 255)
                    { red = 255; }
                    else if (red < 0)
                    { red = 0; }

                    resultBuffer[byteOffset] = (byte)(blue);
                    resultBuffer[byteOffset + 1] = (byte)(green);
                    resultBuffer[byteOffset + 2] = (byte)(red);
                    resultBuffer[byteOffset + 3] = 255;
                }
            }

            Bitmap resultBitmap = new Bitmap(sourceBitmap.Width, sourceBitmap.Height);

            BitmapData resultData = resultBitmap.LockBits(new Rectangle(0, 0,
                                     resultBitmap.Width, resultBitmap.Height),
                                                      ImageLockMode.WriteOnly,
                                                 PixelFormat.Format32bppArgb);

            Marshal.Copy(resultBuffer, 0, resultData.Scan0, resultBuffer.Length);
            resultBitmap.UnlockBits(resultData);

            return resultBitmap;
        }

        public static Bitmap ConvolutionFilter(Bitmap sourceBitmap,
                                                double[,] xFilterMatrix,
                                                double[,] yFilterMatrix,
                                                      double factor = 1,
                                                           int bias = 0,
                                                 bool grayscale = false)
        {
            BitmapData sourceData = sourceBitmap.LockBits(new Rectangle(0, 0,
                                     sourceBitmap.Width, sourceBitmap.Height),
                                                       ImageLockMode.ReadOnly,
                                                  PixelFormat.Format32bppArgb);

            byte[] pixelBuffer = new byte[sourceData.Stride * sourceData.Height];
            byte[] resultBuffer = new byte[sourceData.Stride * sourceData.Height];

            Marshal.Copy(sourceData.Scan0, pixelBuffer, 0, pixelBuffer.Length);
            sourceBitmap.UnlockBits(sourceData);

            if (grayscale == true)
            {
                float rgb = 0;

                for (int k = 0; k < pixelBuffer.Length; k += 4)
                {
                    rgb = pixelBuffer[k] * 0.11f;
                    rgb += pixelBuffer[k + 1] * 0.59f;
                    rgb += pixelBuffer[k + 2] * 0.3f;

                    pixelBuffer[k] = (byte)rgb;
                    pixelBuffer[k + 1] = pixelBuffer[k];
                    pixelBuffer[k + 2] = pixelBuffer[k];
                    pixelBuffer[k + 3] = 255;
                }
            }

            double blueX = 0.0;
            double greenX = 0.0;
            double redX = 0.0;

            double blueY = 0.0;
            double greenY = 0.0;
            double redY = 0.0;

            double blueTotal = 0.0;
            double greenTotal = 0.0;
            double redTotal = 0.0;

            int filterOffset = 1;
            int calcOffset = 0;

            int byteOffset = 0;

            for (int offsetY = filterOffset; offsetY <
                sourceBitmap.Height - filterOffset; offsetY++)
            {
                for (int offsetX = filterOffset; offsetX <
                    sourceBitmap.Width - filterOffset; offsetX++)
                {
                    blueX = greenX = redX = 0;
                    blueY = greenY = redY = 0;

                    blueTotal = greenTotal = redTotal = 0.0;

                    byteOffset = offsetY *
                                 sourceData.Stride +
                                 offsetX * 4;

                    for (int filterY = -filterOffset;
                        filterY <= filterOffset; filterY++)
                    {
                        for (int filterX = -filterOffset;
                            filterX <= filterOffset; filterX++)
                        {
                            calcOffset = byteOffset +
                                         (filterX * 4) +
                                         (filterY * sourceData.Stride);

                            blueX += (double)(pixelBuffer[calcOffset]) *
                                      xFilterMatrix[filterY + filterOffset,
                                              filterX + filterOffset];

                            greenX += (double)(pixelBuffer[calcOffset + 1]) *
                                      xFilterMatrix[filterY + filterOffset,
                                              filterX + filterOffset];

                            redX += (double)(pixelBuffer[calcOffset + 2]) *
                                      xFilterMatrix[filterY + filterOffset,
                                              filterX + filterOffset];

                            blueY += (double)(pixelBuffer[calcOffset]) *
                                      yFilterMatrix[filterY + filterOffset,
                                              filterX + filterOffset];

                            greenY += (double)(pixelBuffer[calcOffset + 1]) *
                                      yFilterMatrix[filterY + filterOffset,
                                              filterX + filterOffset];

                            redY += (double)(pixelBuffer[calcOffset + 2]) *
                                      yFilterMatrix[filterY + filterOffset,
                                              filterX + filterOffset];
                        }
                    }

                    blueTotal = Math.Sqrt((blueX * blueX) + (blueY * blueY));
                    greenTotal = Math.Sqrt((greenX * greenX) + (greenY * greenY));
                    redTotal = Math.Sqrt((redX * redX) + (redY * redY));

                    if (blueTotal > 255)
                    { blueTotal = 255; }
                    else if (blueTotal < 0)
                    { blueTotal = 0; }

                    if (greenTotal > 255)
                    { greenTotal = 255; }
                    else if (greenTotal < 0)
                    { greenTotal = 0; }

                    if (redTotal > 255)
                    { redTotal = 255; }
                    else if (redTotal < 0)
                    { redTotal = 0; }

                    resultBuffer[byteOffset] = (byte)(blueTotal);
                    resultBuffer[byteOffset + 1] = (byte)(greenTotal);
                    resultBuffer[byteOffset + 2] = (byte)(redTotal);
                    resultBuffer[byteOffset + 3] = 255;
                }
            }

            Bitmap resultBitmap = new Bitmap(sourceBitmap.Width, sourceBitmap.Height);

            BitmapData resultData = resultBitmap.LockBits(new Rectangle(0, 0,
                                     resultBitmap.Width, resultBitmap.Height),
                                                      ImageLockMode.WriteOnly,
                                                  PixelFormat.Format32bppArgb);

            Marshal.Copy(resultBuffer, 0, resultData.Scan0, resultBuffer.Length);
            resultBitmap.UnlockBits(resultData);

            return resultBitmap;
        }

        public static Bitmap Laplacian3x3Filter(Bitmap sourceBitmap,
                                                    bool grayscale = true)
        {
            Bitmap resultBitmap = Filters.ConvolutionFilter(sourceBitmap,
                                    Matrix.Laplacian3x3, 1.0, 0, grayscale);

            return resultBitmap;
        }

        public static Bitmap Laplacian5x5Filter(Bitmap sourceBitmap,
                                                    bool grayscale = true)
        {
            Bitmap resultBitmap = Filters.ConvolutionFilter(sourceBitmap,
                                    Matrix.Laplacian5x5, 1.0, 0, grayscale);

            return resultBitmap;
        }

        public static Bitmap LaplacianOfGaussianFilter(Bitmap sourceBitmap)
        {
            Bitmap resultBitmap = Filters.ConvolutionFilter(sourceBitmap,
                                  Matrix.LaplacianOfGaussian, 1.0, 0, true);

            return resultBitmap;
        }

        public static Bitmap Laplacian3x3OfGaussian3x3Filter(Bitmap sourceBitmap)
        {
            Bitmap resultBitmap = Filters.ConvolutionFilter(sourceBitmap,
                                   Matrix.Gaussian3x3, 1.0 / 16.0, 0, true);

            resultBitmap = Filters.ConvolutionFilter(resultBitmap,
                                 Matrix.Laplacian3x3, 1.0, 0, false);

            return resultBitmap;
        }

        public static Bitmap Laplacian3x3OfGaussian5x5Filter1(Bitmap sourceBitmap)
        {
            Bitmap resultBitmap = Filters.ConvolutionFilter(sourceBitmap,
                             Matrix.Gaussian5x5Type1, 1.0 / 159.0, 0, true);

            resultBitmap = Filters.ConvolutionFilter(resultBitmap,
                                 Matrix.Laplacian3x3, 1.0, 0, false);

            return resultBitmap;
        }

        public static Bitmap Laplacian3x3OfGaussian5x5Filter2(Bitmap sourceBitmap)
        {
            Bitmap resultBitmap = Filters.ConvolutionFilter(sourceBitmap,
                             Matrix.Gaussian5x5Type2, 1.0 / 256.0, 0, true);

            resultBitmap = Filters.ConvolutionFilter(resultBitmap,
                                 Matrix.Laplacian3x3, 1.0, 0, false);

            return resultBitmap;
        }

        public static Bitmap Laplacian5x5OfGaussian3x3Filter(Bitmap sourceBitmap)
        {
            Bitmap resultBitmap = Filters.ConvolutionFilter(sourceBitmap,
                                   Matrix.Gaussian3x3, 1.0 / 16.0, 0, true);

            resultBitmap = Filters.ConvolutionFilter(resultBitmap,
                                 Matrix.Laplacian5x5, 1.0, 0, false);

            return resultBitmap;
        }

        public static Bitmap Laplacian5x5OfGaussian5x5Filter1(Bitmap sourceBitmap)
        {
            Bitmap resultBitmap = Filters.ConvolutionFilter(sourceBitmap,
                             Matrix.Gaussian5x5Type1, 1.0 / 159.0, 0, true);

            resultBitmap = Filters.ConvolutionFilter(resultBitmap,
                                 Matrix.Laplacian5x5, 1.0, 0, false);

            return resultBitmap;
        }

        public static Bitmap Laplacian5x5OfGaussian5x5Filter2(Bitmap sourceBitmap)
        {
            Bitmap resultBitmap = Filters.ConvolutionFilter(sourceBitmap,
                                                   Matrix.Gaussian5x5Type2,
                                                     1.0 / 256.0, 0, true);

            resultBitmap = Filters.ConvolutionFilter(resultBitmap,
                                 Matrix.Laplacian5x5, 1.0, 0, false);

            return resultBitmap;
        }

        public static Bitmap Sobel3x3Filter(Bitmap sourceBitmap,
                                                bool grayscale = true)
        {
            Bitmap resultBitmap = Filters.ConvolutionFilter(sourceBitmap,
                                                 Matrix.Sobel3x3Horizontal,
                                                   Matrix.Sobel3x3Vertical,
                                                        1.0, 0, grayscale);

            return resultBitmap;
        }

        public static Bitmap PrewittFilter(Bitmap sourceBitmap,
                                               bool grayscale = true)
        {
            Bitmap resultBitmap = Filters.ConvolutionFilter(sourceBitmap,
                                               Matrix.Prewitt3x3Horizontal,
                                                 Matrix.Prewitt3x3Vertical,
                                                        1.0, 0, grayscale);

            return resultBitmap;
        }

        public static Bitmap KirschFilter(Bitmap sourceBitmap,
                                              bool grayscale = true)
        {
            Bitmap resultBitmap = Filters.ConvolutionFilter(sourceBitmap,
                                                Matrix.Kirsch3x3Horizontal,
                                                  Matrix.Kirsch3x3Vertical,
                                                        1.0, 0, grayscale);

            return resultBitmap;
        }
    }

    #endregion
  }
}

using System;
using System.IO;
using System.Linq;
using ML.Contracts;
using ML.Core;
using ML.DeepMethods.Algorithms;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using System.Collections.Generic;

namespace ML.CatDogDemo
{
  public static class Utils
  {
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

        int filterOffset = (filterWidth - 1) / 2;
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

  public static class EnumerableExtensions
  {
    public static IEnumerable<T> Shuffle<T>(this IEnumerable<T> source, int seed)
    {
      return source.Shuffle(new Random(seed));
    }

    public static IEnumerable<T> Shuffle<T>(this IEnumerable<T> source, Random rng)
    {
      if (source == null) throw new ArgumentNullException("source");
      if (rng == null) throw new ArgumentNullException("rng");

      return source.ShuffleIterator(rng);
    }

    private static IEnumerable<T> ShuffleIterator<T>(
        this IEnumerable<T> source, Random rng)
    {
      var buffer = source.ToList();
      for (int i = 0; i < buffer.Count; i++)
      {
        int j = rng.Next(i, buffer.Count);
        yield return buffer[j];

        buffer[j] = buffer[i];
      }
    }
  }
}

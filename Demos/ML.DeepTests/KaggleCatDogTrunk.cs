using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.IO;
using System.Linq;
using ML.Core;
using ML.DeepMethods.Algorithms;

namespace ML.DeepTests
{
  public class KaggleCatDogTrunk : Runner
  {
    public const int    TRAIN_COUNT  = 10000;
    public const int    TEST_COUNT   = 2500;
    public const string SRC_IMG_FILE = "{0}.{1}.jpg";
    public const string CAT_PREFIX   = "cat.";
    public const string DOG_PREFIX   = "dog.";

    private Dictionary<int, double[]> m_Marks = new Dictionary<int, double[]>
    {
      { 0, new[] { 1.0D, 0.0D } }, // cat
      { 1, new[] { 0.0D, 1.0D } }  // dog
    };
    private Dictionary<int, Class> m_Classes = new Dictionary<int, Class>
    {
      { 0, new Class("cat", 0) },
      { 1, new Class("dog", 1) }
    };

    public override string SrcMark    { get { return "kaggle"; } }
    public override string DataPath   { get { return RootPath+@"\data\cat-dog"; }}
    public override string OutputPath { get { return RootPath+@"\output\cat-dog"; }}

    protected override BackpropAlgorithm CreateAlgorithm()
    {
      return Examples.CreateKaggleCatOrDogDemo1_SEALED();
    }

    public virtual int NormImgSize { get { return 32; } }

    #region Export

    protected override void Export()
    {
      throw new NotSupportedException();
    }

    #endregion

    #region Load

    protected override void Load()
    {
      // preload all
      Console.WriteLine("preload all data...");
      var dataset = new MultiRegressionSample<double[][,]>();
      loadData(TrainPath, dataset);

      //var cnt = 0;
      //foreach (var data in dataset)
      //  exportImageData(data.Key, cnt++);

      Console.WriteLine("shuffling data...");
      Utils.Shuffle(ref dataset);

      // train
      Console.WriteLine("load train data...");
      loadTrain(dataset);

      // test
      Console.WriteLine("load test data...");
      loadTest(dataset);
    }

    private void loadData(string path, MultiRegressionSample<double[][,]> sample)
    {
      sample.Clear();

      var dir = new DirectoryInfo(path);
      var loaded = 0;
      var total = dir.GetFiles().Length;

      int c = 0;
      foreach (var file in dir.EnumerateFiles())
      {
        var data = LoadFile(file.FullName);
        //Utils.ExportImageData(data, @"F:\Work\science\Machine learning\data\cat-dog\train\"+(c++)+".png");

        double[] mark;
        if (file.Name.StartsWith(CAT_PREFIX))
          mark = m_Marks[0];
        else if (file.Name.StartsWith(DOG_PREFIX))
          mark = m_Marks[1];
        else
          throw new MLException("Unknown file");

        sample.Add(data, mark);
        loaded++;
        if (loaded % 1000 == 0)
          Console.Write("\rloaded: {0} of {1}        ", loaded, total);
      };

      Console.WriteLine("\nLoaded files from: {0}", path);
    }

    protected virtual double[][,] LoadFile(string fpath)
    {
      using (var image = (Bitmap)Image.FromFile(fpath))
      {
        var w = image.Width;
        var h = image.Height;
        var s = Math.Min(w, h);

        // crop image to center square size
        // and normalize image to NORM_IMG_SIZE x NORM_IMG_SIZE

        using (var normImage = new Bitmap(NormImgSize, NormImgSize))
        {
          using (var gr = Graphics.FromImage(normImage))
          {
            gr.InterpolationMode  = InterpolationMode.HighQualityBicubic;
            gr.CompositingQuality = CompositingQuality.HighQuality;
            gr.SmoothingMode      = SmoothingMode.AntiAlias;

            gr.DrawImage(image,
                         new Rectangle(0, 0, NormImgSize, NormImgSize),
                         new Rectangle((w - s) / 2, (h - s) / 2, s, s),
                         GraphicsUnit.Pixel);
          }

          // digitize images

          var result = new double[3][,];
          for (int i=0; i<3; i++)
            result[i] = new double[NormImgSize, NormImgSize];

          for (var y=0; y<NormImgSize; y++)
          for (var x=0; x<NormImgSize; x++)
          {
            var pixel = normImage.GetPixel(x, y);
            result[0][y, x] = pixel.R/255.0D;
            result[1][y, x] = pixel.G/255.0D;
            result[2][y, x] = pixel.B/255.0D;
          }

          return result;
        }
      }
    }

    private void loadTrain(MultiRegressionSample<double[][,]> dataset)
    {
      m_TrainingSet = dataset.Subset(0, TRAIN_COUNT);
    }

    private void loadTest(MultiRegressionSample<double[][,]> dataset)
    {
      m_TestingSet = dataset.Subset(TRAIN_COUNT, TEST_COUNT);
    }

    #endregion

    #region Train

    protected override void Train()
    {
      var tstart = DateTime.Now;
      var now = DateTime.Now;

      Alg.EpochEndedEvent += (o, e) =>
                             {
                               Utils.HandleClassificationEpochEnded(Alg, m_TestingSet, m_ValidationSet, m_Classes.Values.ToArray(), OutputPath);
                               tstart = DateTime.Now;
                             };
      Alg.BatchEndedEvent += (o, e) =>
                             {
                               Utils.HandleBatchEnded(Alg, m_TrainingSet.Count, tstart);
                             };

      Console.WriteLine();
      Console.WriteLine("Training started at {0}", now);
      Alg.Train(m_TrainingSet);

      Console.WriteLine("\n--------- ELAPSED TRAIN ----------" + (DateTime.Now-now).TotalMilliseconds);
    }

    #endregion

    #region Test

    protected override void Test()
    {
      throw new NotSupportedException();
    }

    #endregion
  }

  public class KaggleCatDogTrunk_BlackWhite : KaggleCatDogTrunk
  {
    public override string OutputPath { get { return RootPath+@"\output\cat-dog-blackwhite"; }}

    protected override BackpropAlgorithm CreateAlgorithm()
    {
      return Examples.CreateKaggleCatOrDogBlackWhiteDemo1_Pretrained(@"C:\ML\output\cat-dog-blackwhite\_pretrained\___cn_e75_p28.28.mld");
    }

    public override int NormImgSize { get { return 48; } }

    protected override double[][,] LoadFile(string fpath)
    {
      var image = (Bitmap)Image.FromFile(fpath);
      var w = image.Width;
      var h = image.Height;
      var s = Math.Min(w, h);

      // crop image to center square size
      // and normalize image to NORM_IMG_SIZE x NORM_IMG_SIZE

      using (var normImage = new Bitmap(NormImgSize, NormImgSize))
      {
        using (var gr = Graphics.FromImage(normImage))
        {
          gr.InterpolationMode  = InterpolationMode.HighQualityBicubic;
          gr.CompositingQuality = CompositingQuality.HighQuality;
          gr.SmoothingMode      = SmoothingMode.AntiAlias;

          gr.DrawImage(image,
                       new Rectangle(0, 0, NormImgSize, NormImgSize),
                       new Rectangle((w - s) / 2, (h - s) / 2, s, s),
                       GraphicsUnit.Pixel);
        }

        // digitize images

        var result = new double[1][,] { new double[NormImgSize, NormImgSize] };

        for (var y=0; y<NormImgSize; y++)
        for (var x=0; x<NormImgSize; x++)
        {
          var pixel = normImage.GetPixel(x, y);
          var level = (pixel.R + pixel.G + pixel.B) / (3*255.0D);
          result[0][y, x] = level;
        }

        return result;
      }
    }

  }

  public class KaggleCatDogTrunk_Filters : KaggleCatDogTrunk
  {
    public override string OutputPath { get { return RootPath+@"\output\cat-dog-filters"; }}

    protected override BackpropAlgorithm CreateAlgorithm()
    {
      //return Examples.CreateKaggleCatOrDogFiltersDemo1();
      return Examples.CreateKaggleCatOrDogFiltersDemo1_Pretrained(@"C:\ML\output\cat-dog-filters\_pretrained\cn_e20_p36.64.mld");
    }

    public override int NormImgSize { get { return 48; } }

    protected override double[][,] LoadFile(string fpath)
    {
      using (var image = (Bitmap)Image.FromFile(fpath))
      using (var filtImage = Utils.Filters.Sobel3x3Filter(image))
      using (var normImage = new Bitmap(NormImgSize, NormImgSize))
      using (var normFiltImage = new Bitmap(NormImgSize, NormImgSize))
      {
        var w = image.Width;
        var h = image.Height;
        var s = Math.Min(w, h);

        // crop image to center square size
        // and normalize image to NORM_IMG_SIZE x NORM_IMG_SIZE

        using (var gr = Graphics.FromImage(normImage))
        {
          gr.InterpolationMode  = InterpolationMode.HighQualityBicubic;
          gr.CompositingQuality = CompositingQuality.HighQuality;
          gr.SmoothingMode      = SmoothingMode.AntiAlias;

          gr.DrawImage(image,
                       new Rectangle(0, 0, NormImgSize, NormImgSize),
                       new Rectangle((w - s) / 2, (h - s) / 2, s, s),
                       GraphicsUnit.Pixel);
        }
        using (var gr = Graphics.FromImage(normFiltImage))
        {
          gr.InterpolationMode  = InterpolationMode.HighQualityBicubic;
          gr.CompositingQuality = CompositingQuality.HighQuality;
          gr.SmoothingMode      = SmoothingMode.AntiAlias;

          gr.DrawImage(filtImage,
                       new Rectangle(0, 0, NormImgSize, NormImgSize),
                       new Rectangle((w - s) / 2, (h - s) / 2, s, s),
                       GraphicsUnit.Pixel);
        }

        // digitize images

        var result = new double[2][,]
        {
          new double[NormImgSize, NormImgSize],
          new double[NormImgSize, NormImgSize]
        };

        // grayscale
        for (var y=0; y<NormImgSize; y++)
        for (var x=0; x<NormImgSize; x++)
        {
          var pixel = normImage.GetPixel(x, y);
          var level = (pixel.R + pixel.G + pixel.B) / (3*255.0D);
          result[0][y, x] = level;
        }

        // filter
        for (var y=0; y<NormImgSize; y++)
        for (var x=0; x<NormImgSize; x++)
        {
          var pixel = normFiltImage.GetPixel(x, y);
          var level = (pixel.R + pixel.G + pixel.B) / (3*255.0D);
          result[1][y, x] = level;
        }

        return result;
      }
    }
  }

}

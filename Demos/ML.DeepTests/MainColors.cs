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
  public class MainColors : Runner
  {
    public const string IMG_PREFIX = "img_";
    public const int    TRAIN_PCT = 70;


    public override string SrcMark    { get { return "original"; } }
    public override string DataPath   { get { return RootPath + @"\data\main-colors"; } }
    public override string OutputPath { get { return RootPath + @"\output\main-colors"; } }

    protected override BackpropAlgorithm CreateAlgorithm()
    {
      return Examples.CreateMainColorsDemo1_Pretrain(@"F:\Work\science\Machine learning\data\main-colors-test\pretrain\original\net.mld");
      //return Examples.CreateMainColorsDemo1();
    }

    public virtual int NormImgSize { get { return 48; } }

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

      var marks = new Dictionary<int, double[]>();
      var lpath = Path.Combine(path, "labels.csv");
      using (var lfile = File.Open(lpath, FileMode.Open, FileAccess.Read))
      using (var reader = new StreamReader(lfile))
      {
        reader.ReadLine(); // header
        while (true)
        {
          var line = reader.ReadLine();
          if (string.IsNullOrWhiteSpace(line)) break;

          var segs = line.Split(',');
          var id = int.Parse(segs[0]);
          var len = segs.Length-1;
          var mark = new double[len];
          for (int i=0; i<len; i++)
            mark[i] = int.Parse(segs[i+1])/255.0D;

          marks[id] = mark;
        }
      }

      foreach (var file in dir.EnumerateFiles().Where(f => f.Name.StartsWith(IMG_PREFIX)))
      {
        var fname = Path.GetFileNameWithoutExtension(file.Name);
        var id = int.Parse(fname.Substring(IMG_PREFIX.Length));
        var data = loadFile(file.FullName);
        Utils.ExportImageData(data, @"F:\Work\science\Machine learning\data\main-colors\train\1.png");

        sample.Add(data, marks[id]);
        loaded++;
        if (loaded % 1000 == 0)
          Console.Write("\rloaded: {0} of {1}        ", loaded, total);
      };

      Console.WriteLine("\nLoaded files from: {0}", path);
    }

    private double[][,] loadFile(string fpath)
    {
      var image = (Bitmap)Image.FromFile(fpath);
      var w = image.Width;
      var h = image.Height;
      var s = Math.Min(w, h);

      // crop image to center square size
      // and normalize image to NORM_IMG_SIZE x NORM_IMG_SIZE

      var normImage = new Bitmap(NormImgSize, NormImgSize);
      using (var gr = Graphics.FromImage(normImage))
      {
        gr.InterpolationMode = InterpolationMode.HighQualityBicubic;
        gr.CompositingQuality = CompositingQuality.HighQuality;
        gr.SmoothingMode = SmoothingMode.AntiAlias;

        gr.DrawImage(image,
                     new Rectangle(0, 0, NormImgSize, NormImgSize),
                     new Rectangle((w - s) / 2, (h - s) / 2, s, s),
                     GraphicsUnit.Pixel);
      }

      // digitize images

      var result = new double[3][,];
      for (int i = 0; i < 3; i++)
        result[i] = new double[NormImgSize, NormImgSize];

      int x, y;
      for (y = 0; y < NormImgSize; y++)
        for (x = 0; x < NormImgSize; x++)
        {
          var pixel = normImage.GetPixel(x, y);
          result[0][y, x] = pixel.R / 255.0D;
          result[1][y, x] = pixel.G / 255.0D;
          result[2][y, x] = pixel.B / 255.0D;
        }

      return result;
    }

    private void loadTrain(MultiRegressionSample<double[][,]> dataset)
    {
      var trainCount = dataset.Count * TRAIN_PCT / 100;
      m_TrainingSet = dataset.Subset(0, trainCount);
    }

    private void loadTest(MultiRegressionSample<double[][,]> dataset)
    {
      var trainCount = dataset.Count * TRAIN_PCT / 100;
      var testCount  = dataset.Count - trainCount;
      m_TestingSet   = dataset.Subset(trainCount, testCount);
    }

    #endregion

    #region Train

    protected override void Train()
    {
      var tstart = DateTime.Now;
      var now = DateTime.Now;

      Alg.EpochEndedEvent += (o, e) =>
                             {
                               Utils.HandleRegressionEpochEnded(Alg, m_TestingSet, m_ValidationSet, OutputPath);
                               tstart = DateTime.Now;
                             };
      Alg.BatchEndedEvent += (o, e) =>
                             {
                               Utils.HandleBatchEnded(Alg, m_TrainingSet.Count, tstart);
                             };

      Console.WriteLine();
      Console.WriteLine("Training started at {0}", now);
      Alg.Train(m_TrainingSet);

      Console.WriteLine("\n--------- ELAPSED TRAIN ----------" + (DateTime.Now - now).TotalMilliseconds);
    }

    #endregion

    #region Test

    protected override void Test()
    {
      throw new NotSupportedException();
    }

    #endregion
  }
}

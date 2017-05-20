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
    public const int    TEST_COUNT   = 2000;
    public const string SRC_IMG_FILE = "{0}.{1}.jpg";
    public const string CAT_PREFIX   = "cat.";
    public const string DOG_PREFIX   = "dog.";

    private Dictionary<int, double[]> m_Classes = new Dictionary<int, double[]>
    {
      { 0, new[] { 1.0D, 0.0D } }, // cat
      { 1, new[] { 0.0D, 1.0D } }  // dog
    };

    public override string SrcMark    { get { return "kaggle"; } }
    public override string DataPath   { get { return RootPath+@"\data\cat-dog"; }}
    public override string OutputPath { get { return RootPath+@"\output\cat-dog"; }}

    protected override BackpropAlgorithm CreateAlgorithm()
    {
      return Examples.CreateKaggleCatOrDogDemo1();
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
      //var path = @"C:\Users\User\Desktop\science\Machine learning\data\cat-dog\train\kaggle";
      //var tezt = new ClassifiedSample<double[][,]>();
      //loadData(path, tezt);
      //var alg = CreateAlgorithm(tezt);
      //var net = alg.Net;
      //net.IsTraining = false;
      //var valid = tezt.Subset(0, 2000);
      //int errors = 0;
      //foreach (var data in valid)
      //{
      //  var res = alg.Classify(data.Key);
      //  if (res != data.Value)
      //    errors += 1;
      //}
      //Console.WriteLine(errors);
      //Console.ReadLine();



      // preload all
      Console.WriteLine("preload all data...");
      var dataset = new MultiRegressionSample<double[][,]>();
      loadData(TrainPath, dataset);

      //var cnt = 0;
      //foreach (var data in dataset)
      //  exportImageData(data.Key, cnt++);

      Console.WriteLine("shuffling data...");
      shuffle(ref dataset);

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

      foreach (var file in dir.EnumerateFiles())
      {
        var data = loadFile(file.FullName);

        double[] mark;
        if (file.Name.StartsWith(CAT_PREFIX))
          mark = m_Classes[0];
        else if (file.Name.StartsWith(DOG_PREFIX))
          mark = m_Classes[1];
        else
          throw new MLException("Unknown file");

        lock (sample)
        {
          sample.Add(data, mark);
          loaded++;
          if (loaded % 1000 == 0)
            Console.Write("\rloaded: {0} of {1}        ", loaded, total);
        }
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
                     new Rectangle((w-s)/2, (h-s)/2, s, s),
                     GraphicsUnit.Pixel);
      }

      // digitize images

      var result = new double[3][,];
      for (int i=0; i<3; i++)
        result[i] = new double[NormImgSize, NormImgSize];

      int x,y;
      for (y=0; y<NormImgSize; y++)
      for (x=0; x<NormImgSize; x++)
      {
        var pixel = normImage.GetPixel(x, y);
        result[0][y, x] = pixel.R/255.0D;
        result[1][y, x] = pixel.G/255.0D;
        result[2][y, x] = pixel.B/255.0D;
      }

      return result;
    }

    private void shuffle(ref MultiRegressionSample<double[][,]> sample)
    {
      var result = new MultiRegressionSample<double[][,]>();

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

    private void loadTrain(MultiRegressionSample<double[][,]> dataset)
    {
      m_TrainingSet = dataset.Subset(0, TRAIN_COUNT);
    }

    private void exportImageData(double[][,] data, int cnt)
    {
      using (var ofile = File.Open(string.Format(@"C:\Users\User\Desktop\science\Machine learning\data\cat-dog\train\file_{0}.png", cnt), FileMode.Create, FileAccess.Write))
      {
        var image = new Bitmap(NormImgSize, NormImgSize);

        for (int y = 0; y < NormImgSize; y++)
        for (int x = 0; x < NormImgSize; x++)
        {
          image.SetPixel(x, y, Color.FromArgb((int)(data[0][y, x]*255), (int)(data[1][y, x]*255), (int)(data[2][y, x]*255)));
        }

        image.Save(ofile, System.Drawing.Imaging.ImageFormat.Png);

        ofile.Flush();
      }
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
                               Utils.HandleEpochEnded(Alg, m_TestingSet, m_ValidationSet, OutputPath);
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
}

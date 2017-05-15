using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using ML.Core;
using System.Threading.Tasks;

namespace ML.DeepTests
{
  public class KaggleCatDogTrunk : Runner
  {
    public const int    TRAIN_COUNT = 10000;
    public const int    TEST_COUNT  = 2000;
    public const int    NORM_IMG_SIZE = 32;
    public const string SRC_IMG_FILE = "{0}.{1}.jpg";
    public const string CAT_PREFIX = "cat.";
    public const string DOG_PREFIX = "dog.";

    private Dictionary<int, Class> m_Classes = new Dictionary<int, Class>
    {
      { 0, new Class("cat", 0) },
      { 1, new Class("dog", 1) }
    };

    public override string SrcMark    { get { return "kaggle"; } }
    public override string DataPath   { get { return RootPath+@"\data\cat-dog"; }}
    public override string OutputPath { get { return RootPath+@"\output\cat-dog"; }}

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
      var dataset = new ClassifiedSample<double[][,]>();
      loadData(TrainPath, dataset);

      Console.WriteLine("shuffling data...");
      shuffle(ref dataset);

      // train
      Console.WriteLine("load train data...");
      loadTrain(dataset);

      // test
      Console.WriteLine("load test data...");
      loadTest(dataset);
    }

    private void loadData(string path, ClassifiedSample<double[][,]> sample)
    {
      sample.Clear();

      var dir = new DirectoryInfo(path);
      var loaded = 0;
      var total = dir.GetFiles().Length;

      foreach (var file in dir.EnumerateFiles())
      {
        var data = loadFile(file.FullName);

        Class cls;
        if (file.Name.StartsWith(CAT_PREFIX))
          cls = m_Classes[0];
        else if (file.Name.StartsWith(DOG_PREFIX))
          cls = m_Classes[1];
        else
          throw new MLException("Unknown file");

        lock (sample)
        {
          sample.Add(data, cls);
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

      var normImage = new Bitmap(NORM_IMG_SIZE, NORM_IMG_SIZE);
      using (var gr = Graphics.FromImage(normImage))
      {
        gr.DrawImage(image,
                     new Rectangle(0, 0, NORM_IMG_SIZE, NORM_IMG_SIZE),
                     new Rectangle((w-s)/2, (h-s)/2, s, s),
                     GraphicsUnit.Pixel);
      }

      // digitize images

      var result = new double[3][,];
      for (int i=0; i<3; i++)
        result[i] = new double[NORM_IMG_SIZE, NORM_IMG_SIZE];

      int x,y;
      for (y=0; y<NORM_IMG_SIZE; y++)
      for (x=0; x<NORM_IMG_SIZE; x++)
      {
        var pixel = normImage.GetPixel(x, y);
        result[0][y, x] = pixel.R/255.0D;
        result[1][y, x] = pixel.G/255.0D;
        result[2][y, x] = pixel.B/255.0D;
      }

      return result;
    }

    private void shuffle(ref ClassifiedSample<double[][,]> sample)
    {
      var result = new ClassifiedSample<double[][,]>();

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

    private void loadTrain(ClassifiedSample<double[][,]> dataset)
    {
      m_TrainingSet = dataset.Subset(0, TRAIN_COUNT);
    }

    private void loadTest(ClassifiedSample<double[][,]> dataset)
    {
      m_TestingSet = dataset.Subset(TRAIN_COUNT, TEST_COUNT);
    }

    #endregion

    #region Train

    protected override void Train()
    {
      var tstart = DateTime.Now;
      var now = DateTime.Now;

      Alg = Examples.CreateKaggleCatOrDogDemo_Pretrained(m_TrainingSet);
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
      Alg.Train();

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

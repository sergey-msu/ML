using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using ML.Core;
using System.Threading.Tasks;
using ML.DeepMethods.Algorithms;

namespace ML.DeepTests
{
  public class KaggleCatDog : Runner
  {
    public const int    NORM_IMG_SIZE = 32;
    public const string SRC_IMG_FILE = "{0}.{1}.jpg";
    public const string CAT_PREFIX = "cat.";
    public const string DOG_PREFIX = "dog.";

    private List<double[][,]>      m_Test = new List<double[][,]>();
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

    #region Export

    protected override void Export()
    {
      throw new NotSupportedException();
    }

    #endregion

    #region Load

    protected override void Load()
    {
      // train
      Console.WriteLine("load train data...");
      loadTrain(TrainPath, m_TrainingSet);
      Console.WriteLine("shuffling train data...");
      Utils.Shuffle(ref m_TrainingSet);

      // test
      //Console.WriteLine("load test data...");
      //loadTest(TestPath, m_Test);
    }

    private void loadTrain(string path, MultiRegressionSample<double[][,]> sample)
    {
      sample.Clear();

      var dir = new DirectoryInfo(path);
      var loaded = 0;
      var total = dir.GetFiles().Length;

      Parallel.ForEach(dir.EnumerateFiles(), file =>
      {
        var data = loadFile(file.FullName);

        double[] mark;
        if (file.Name.StartsWith(CAT_PREFIX))
          mark = m_Marks[0];
        else if (file.Name.StartsWith(DOG_PREFIX))
          mark = m_Marks[1];
        else
          throw new MLException("Unknown file");

        lock (sample)
        {
          sample.Add(data, mark);
          loaded++;
          if (loaded % 1000 == 0)
            Console.Write("\rloaded: {0} of {1}        ", loaded, total);
        }
      });

      Console.WriteLine("\nLoaded files from: {0}", path);
    }

    //private void loadTest(string path, List<double[][,]> sample)
    //{
    //  sample.Clear();
    //
    //  var dir = new DirectoryInfo(path);
    //  foreach (var file in dir.EnumerateFiles())
    //  {
    //     var data = loadFile(file.FullName);
    //     sample.Add(data);
    //  }
    //
    //  Console.WriteLine("Loaded files from: {0}", path);
    //}

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

    #endregion

    #region Train

    protected override void Train()
    {
      var tstart = DateTime.Now;
      var now = DateTime.Now;

      Alg.EpochEndedEvent += (o, e) =>
                             {
                               Utils.HandleClassificationEpochEnded(Alg, null, m_ValidationSet, m_Classes.Values.ToArray(), OutputPath); // no labeled testing set in Kaggle :(
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
      throw new NotImplementedException(); // TODO
    }

    #endregion
  }
}

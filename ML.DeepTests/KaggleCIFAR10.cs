using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using ML.Core;
using ML.Utils;
using ML.DeepMethods.Algorithms;

namespace ML.DeepTests
{
  public class KaggleCIFAR10 : Runner
  {
    const int IMG_SIZE = 32;
    const string Cifar10_IMG_FILE = "{0}.png";

    private List<double[,,]> m_Test = new List<double[,,]>();
    private Dictionary<int, Class> m_Classes = new Dictionary<int, Class>
    {
      { 0, new Class("airplane",   0) },
      { 1, new Class("automobile", 1) },
      { 2, new Class("bird",       2) },
      { 3, new Class("cat",        3) },
      { 4, new Class("deer",       4) },
      { 5, new Class("dog",        5) },
      { 6, new Class("frog",       6) },
      { 7, new Class("horse",      7) },
      { 8, new Class("ship",       8) },
      { 9, new Class("truck",      9) },
    };

    public string Cifar10Path   { get { return Root+@"\data\cifar10"; }}
    public string Cifar10Src    { get { return Cifar10Path+@"\src\kaggle"; }}
    public string Cifar10Test   { get { return Cifar10Path+@"\test\kaggle"; }}
    public string Cifar10Train  { get { return Cifar10Path+@"\train\kaggle"; }}
    public override string ResultsFolder { get { return Root+@"\output\cifar10_kaggle"; }}


    protected override void Init()
    {
      base.Init();

      var paths = new []{ Root, Cifar10Path, Cifar10Src, Cifar10Test, Cifar10Train, ResultsFolder };
      foreach (var path in paths)
      {
        if (!Directory.Exists(path))
          Directory.CreateDirectory(path);
      }
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
      var labFilePath = Path.Combine(Cifar10Src, "train.csv");
      loadTrain(Cifar10Train, labFilePath, m_Training);

      // test
      //Console.WriteLine("load test data...");
      //loadTest(Cifar10Test, m_Test);
    }

    private void loadTrain(string path, string lpath, ClassifiedSample<double[,,]> sample)
    {
      sample.Clear();

      using (var lfile = File.Open(lpath, FileMode.Open, FileAccess.Read))
      using (var reader = new StreamReader(lfile))
      {
        reader.ReadLine(); // read label file header

        var dir = new DirectoryInfo(path);
        foreach (var file in dir.EnumerateFiles())
        {
           var data = loadFile(file.FullName);
           var clsName = reader.ReadLine().Split(',')[1];
           var cls = m_Classes.First(c => c.Value.Name.Equals(clsName)).Value;
           sample.Add(data, cls);
        }
      }

      Console.WriteLine("Loaded files from: {0}", path);
    }

    private void loadTest(string path, List<double[,,]> sample)
    {
      sample.Clear();

      var dir = new DirectoryInfo(path);
      foreach (var file in dir.EnumerateFiles())
      {
         var data = loadFile(file.FullName);
         sample.Add(data);
      }

      Console.WriteLine("Loaded files from: {0}", path);
    }

    private double[,,] loadFile(string fpath)
    {
      var image = (Bitmap)Image.FromFile(fpath);
      var result = new double[3, image.Height, image.Width];

      int x,y;
      for (y=0; y<image.Width; y++)
      for (x=0; x<image.Width; x++)
      {
        var pixel = image.GetPixel(x, y);
        result[0, y, x] = pixel.R/255.0D;
        result[1, y, x] = pixel.G/255.0D;
        result[2, y, x] = pixel.B/255.0D;
      }

      return result;
    }

    #endregion

    #region Learning

    protected override void Train()
    {
      // create CNN
      var net = NetworkFactory.CreateCIFAR10Net();
      //net[net.LayerCount-1].ActivationFunction = Registry.ActivationFunctions.ReLU;

      // create algorithm
      var epochs = 100;
      Alg = new BackpropAlgorithm(m_Training, net)
      {
        LossFunction = Registry.LossFunctions.CrossEntropySoftMax,
        EpochCount = epochs,
        LearningRate = 0.005D
      };
      Alg.EpochEndedEvent += (o, e) => Utils.HandleEpochEnded(Alg, m_Training, ResultsFolder); // we do not have public test data in kaggle :(

      // run training process
      var now = DateTime.Now;
      Console.WriteLine();
      Console.WriteLine("Training started at {0}", now);
      Alg.Train();

      Console.WriteLine("--------- ELAPSED TRAIN ----------" + (DateTime.Now-now).TotalMilliseconds);
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

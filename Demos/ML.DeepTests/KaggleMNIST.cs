using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using ML.Core;
using ML.DeepMethods.Algorithms;
using ML.DeepMethods.Models;

namespace ML.DeepTests
{
  public class KaggleMNIST : Runner
  {
    const int IMG_SIZE = 28;
    const string MNIST_IMG_FILE = "kaggle_img_{0}.png";

    private List<double[][,]> m_Test = new List<double[][,]>();
    private Dictionary<int, Class> m_Classes = new Dictionary<int, Class>
    {
      { 0, new Class("Zero",  0) },
      { 1, new Class("One",   1) },
      { 2, new Class("Two",   2) },
      { 3, new Class("Three", 3) },
      { 4, new Class("Four",  4) },
      { 5, new Class("Five",  5) },
      { 6, new Class("Six",   6) },
      { 7, new Class("Seven", 7) },
      { 8, new Class("Eight", 8) },
      { 9, new Class("Nine",  9) },
    };

    public override string SrcMark    { get { return "kaggle"; } }
    public override string DataPath   { get { return RootPath+@"\data\mnist"; }}
    public override string OutputPath { get { return RootPath+@"\output\mnist_kaggle"; }}

    #region Export

    protected override void Export()
    {
      // train
      var objFilePath = Path.Combine(SrcPath, "train.csv");
      exportObjects(objFilePath, TrainPath, true);

      // test
      objFilePath = Path.Combine(SrcPath, "test.csv");
      exportObjects(objFilePath, TrainPath, false);
    }

    private void exportObjects(string fpath, string opath, bool isTrain)
    {
      using (var file = File.Open(fpath, FileMode.Open, FileAccess.Read))
      using (var reader = new StreamReader(file))
      {
        var header = reader.ReadLine();

        int num = 1;
        while (true)
        {
          var str = reader.ReadLine();
          if (string.IsNullOrWhiteSpace(str)) break;

          var data = str.Split(',')
                        .Skip(isTrain ? 1 : 0)
                        .Select(d => int.Parse(d))
                        .ToArray();

          exportImageData(data, opath, num++);

          if (num%100==0)
            Console.WriteLine("Exported: {0}", num);
        }
      }
    }

    private void exportImageData(int[] data, string opath, int counter)
    {
      var oname = Path.Combine(opath, string.Format(MNIST_IMG_FILE, counter));

      using (var ofile = File.Open(oname, FileMode.Create, FileAccess.Write))
      {
        var image = new Bitmap(IMG_SIZE, IMG_SIZE);

        for (int i=0; i<IMG_SIZE*IMG_SIZE; i++)
        {
          var shade = 255-data[i]; // 0 in file means white, 255 - black
                                   // invert 255-* to map to image color
          var x = i%IMG_SIZE;
          var y = i/IMG_SIZE;
          image.SetPixel(x, y, Color.FromArgb(shade, shade, shade));
        }

        image.Save(ofile, ImageFormat.Png);

        ofile.Flush();
      }
    }

    #endregion

    #region Load

    protected override void Load()
    {
      // train
      Console.WriteLine("load train data...");
      var objFilePath = Path.Combine(SrcPath, "train.csv");
      loadSample(objFilePath, m_TrainingSet);

      // test
      Console.WriteLine("load test data...");
      objFilePath = Path.Combine(SrcPath, "test.csv");
      loadTest(objFilePath, m_Test);
    }

    private void loadSample(string ipath, ClassifiedSample<double[][,]> sample)
    {
      sample.Clear();

      using (var ifile = File.Open(ipath, FileMode.Open, FileAccess.Read))
      using (var reader = new StreamReader(ifile))
      {
        var header = reader.ReadLine();

        while (true)
        {
          var str = reader.ReadLine();
          if (string.IsNullOrWhiteSpace(str)) break;

          var raw = str.Split(',')
                       .Select(d => int.Parse(d))
                       .ToArray();

          var label = raw[0];
          var data = new double[1][,] { new double[IMG_SIZE, IMG_SIZE] };

          for (int i=1; i<=IMG_SIZE*IMG_SIZE; i++)
          {
            var shade = raw[i]; // do not invert 255-* because we want to keep logical format: 0=white, 255=black - not image color format!
            var x = (i-1)%IMG_SIZE;
            var y = (i-1)/IMG_SIZE;
            data[0][y, x] = shade/255.0D;
          }
          sample.Add(data, m_Classes[label]);
        }

        Console.WriteLine("Loaded: {0}", ipath);
      }
    }

    private void loadTest(string ipath, List<double[][,]> sample)
    {
      sample.Clear();

      using (var ifile = File.Open(ipath, FileMode.Open, FileAccess.Read))
      using (var reader = new StreamReader(ifile))
      {
        var header = reader.ReadLine();

        while (true)
        {
          var str = reader.ReadLine();
          if (string.IsNullOrWhiteSpace(str)) break;

          var raw = str.Split(',')
                       .Select(d => int.Parse(d))
                       .ToArray();

          var data = new double[1][,] { new double[IMG_SIZE, IMG_SIZE] };
          for (int i=0; i<IMG_SIZE*IMG_SIZE; i++)
          {
            var shade = raw[i]; // do not invert 255-* because we want to keep logical format: 0=white, 255=black - not image color format!
            var x = i%IMG_SIZE;
            var y = i/IMG_SIZE;
            data[0][y, x] = shade/255.0D;
          }
          sample.Add(data);
        }

        Console.WriteLine("Loaded: {0}", ipath);
      }
    }

    #endregion

    #region Train

    protected override void Train()
    {
      Alg = Examples.CreateMNISTSimpleDemo_SEALED(m_TrainingSet);
      Alg.EpochEndedEvent += (o, e) => Utils.HandleEpochEnded(Alg, m_TrainingSet.Subset(0, 10000), m_ValidationSet, OutputPath); // we do not have public test data in kaggle :(

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
      ConvNet lenet1;
      var fpath = Path.Combine(OutputPath, "cn_e26_p0.06.mld");
      //var fpath = @"F:\Work\git\ML\solution\ML.DigitsDemo\lenet1.mld";

      using (var stream = File.Open(fpath, FileMode.Open))
      {
        lenet1 = ConvNet.Deserialize(stream);
      }
      var alg = new BackpropAlgorithm(m_TrainingSet, lenet1);

      var fout = Path.Combine(SrcPath, "result1.csv");
      using (var file = File.Open(fout, FileMode.Create, FileAccess.Write))
      using (var writer = new StreamWriter(file))
      {
        writer.WriteLine("ImageId,Label");

        int num = 1;
        foreach (var data in m_Test)
        {
          var cls = alg.Classify(data);
          writer.WriteLine("{0},{1}", num++, (int)cls.Value);
        }

        writer.Flush();
      }
    }

    #endregion
  }
}

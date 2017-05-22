using System;
using System.Linq;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using ML.Core;
using ML.Utils;
using ML.DeepMethods.Algorithms;

namespace ML.DeepTests
{
  /// <summary>
  /// https://www.cs.toronto.edu/~kriz/cifar.html
  ///
  ///<1 x label><3072 x pixel>
  ///...
  ///<1 x label><3072 x pixel>
  ///
  ///In other words, the first byte is the label of the first image, which is a number in the range 0-9.
  ///The next 3072 bytes are the values of the pixels of the image.
  ///The first 1024 bytes are the red channel values, the next 1024 the green, and the final 1024 the blue.
  /// The values are stored in row-major order, so the first 32 bytes are the red channel values of the first row of the image.
  ///
  ///Each file contains 10000 such 3073-byte "rows" of images, although there is nothing delimiting the rows.
  ///Therefore each file should be exactly 30730000 bytes long.
  /// </summary>
  public class OriginalCIFAR10 : Runner
  {
    const string CIFAR10_IMG_FILE   = "img_{0}.png";
    const string CIFAR10_LABEL_FILE = "labels.csv";

    private Dictionary<int, Class> m_Classes = new Dictionary<int, Class>()
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
    private Dictionary<int, double[]> m_Marks = new Dictionary<int, double[]>
    {
      { 0, new double[] { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 } }, // airplane
      { 1, new double[] { 0, 1, 0, 0, 0, 0, 0, 0, 0, 0 } }, // automobile
      { 2, new double[] { 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 } }, // bird
      { 3, new double[] { 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 } }, // cat
      { 4, new double[] { 0, 0, 0, 0, 1, 0, 0, 0, 0, 0 } }, // deer
      { 5, new double[] { 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 } }, // dog
      { 6, new double[] { 0, 0, 0, 0, 0, 0, 1, 0, 0, 0 } }, // frog
      { 7, new double[] { 0, 0, 0, 0, 0, 0, 0, 1, 0, 0 } }, // horse
      { 8, new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 1, 0 } }, // ship
      { 9, new double[] { 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 } }, // truck
    };

    public override string SrcMark    { get { return "original"; } }
    public override string DataPath   { get { return RootPath+@"\data\cifar10"; }}
    public override string OutputPath { get { return RootPath+@"\output\cifar10_original"; }}

    protected override BackpropAlgorithm CreateAlgorithm()
    {
      return Examples.CreateCIFAR10Demo2();
    }

    protected override void Init()
    {
      base.Init();

      var paths = new []{ RootPath, DataPath, SrcPath, TestPath, TrainPath, OutputPath };
      foreach (var path in paths)
      {
        if (!Directory.Exists(path))
          Directory.CreateDirectory(path);
      }
    }

    #region Export

    protected override void Export()
    {
      // train
      var filePaths = new string[]
      {
        Path.Combine(SrcPath, "data_batch_1.bin"),
        Path.Combine(SrcPath, "data_batch_2.bin"),
        Path.Combine(SrcPath, "data_batch_3.bin"),
        Path.Combine(SrcPath, "data_batch_4.bin"),
        Path.Combine(SrcPath, "data_batch_5.bin")
      };
      exportObjects(filePaths, TrainPath);

      // test
      filePaths = new string[]
      {
        Path.Combine(SrcPath, "test_batch.bin")
      };
      exportObjects(filePaths, TestPath);
    }

    private void exportObjects(string[] fpaths, string opath)
    {
      var lname = Path.Combine(opath, CIFAR10_LABEL_FILE);
      using (var lfile = File.Open(lname, FileMode.Create, FileAccess.Write))
      using (var writer = new StreamWriter(lfile))
      {
        int q = 0;
        foreach (var fpath in fpaths)
        {
          using (var file = File.Open(fpath, FileMode.Open, FileAccess.Read))
          {
            while (true)
            {
              var label = file.ReadByte();
              if (label<0) break;

              var cls  = m_Classes[label];
              var data = new byte[3, 32, 32];

              for (int d = 0; d < 3; d++)
              for (int y = 0; y < 32; y++)
              for (int x = 0; x < 32; x++)
              {
                data[d, y, x] = (byte)file.ReadByte();
              }

              exportImageData(data, opath, q);
              exportLabel(writer, cls, q);

              if ((++q) % 10000 == 0) Console.WriteLine("Exported: {0}", q);
            }
          }
        }
      }
    }

    private void exportImageData(byte[,,] data, string opath, int counter)
    {
      var oname = Path.Combine(opath, string.Format(CIFAR10_IMG_FILE, counter));

      using (var ofile = File.Open(oname, FileMode.Create, FileAccess.Write))
      {
        var image = new Bitmap(32, 32);

        for (int y=0; y<32; y++)
        for (int x=0; x<32; x++)
        {
          image.SetPixel(x, y, Color.FromArgb(data[0,y,x], data[1,y,x], data[2,y,x]));
        }

        image.Save(ofile, ImageFormat.Png);

        ofile.Flush();
      }
    }

    private void exportLabel(StreamWriter writer, Class cls, int counter)
    {
      writer.WriteLine("{0},{1},{2}", counter, cls.Value, cls.Name);
      writer.Flush();
    }

    #endregion

    #region Load

    protected override void Load()
    {
      // train
      Console.WriteLine("load train data...");
      var filePaths = new string[]
      {
        Path.Combine(SrcPath, "data_batch_1.bin"),
        Path.Combine(SrcPath, "data_batch_2.bin"),
        Path.Combine(SrcPath, "data_batch_3.bin"),
        Path.Combine(SrcPath, "data_batch_4.bin"),
        Path.Combine(SrcPath, "data_batch_5.bin")
      };
      loadSample(filePaths, m_TrainingSet);

      // test
      Console.WriteLine("load test data...");
      filePaths = new string[]
      {
        Path.Combine(SrcPath, "test_batch.bin")
      };
      loadSample(filePaths, m_TestingSet);
    }

    private void loadSample(string[] fpaths, MultiRegressionSample<double[][,]> sample)
    {
      foreach (var fpath in fpaths)
      {
        using (var file = File.Open(fpath, FileMode.Open, FileAccess.Read))
        {
          while (true)
          {
            var label = file.ReadByte();
            if (label<0) break;

            var cls = m_Marks[label];

            var data = new double[3][,];
            data[0] = new double[32, 32];
            data[1] = new double[32, 32];
            data[2] = new double[32, 32];

            for (int d = 0; d < 3; d++)
            for (int y = 0; y < 32; y++)
            for (int x = 0; x < 32; x++)
            {
              data[d][y, x] = file.ReadByte()/255.0D;
            }

            sample.Add(data, cls);
          }
        }
      }
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
}

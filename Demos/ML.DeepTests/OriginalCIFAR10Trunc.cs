using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using ML.Core;
using ML.Utils;
using System.Linq;
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
  public class OriginalCIFAR10Trunc : Runner
  {
    const string CIFAR10_IMG_FILE   = "img_{0}.png";
    const string CIFAR10_LABEL_FILE = "labels.csv";

    private Dictionary<int, double[]> m_Marks = new Dictionary<int, double[]>
    {
      { 0, new[] { 1.0D, 0.0D } }, // cat
      { 1, new[] { 0.0D, 1.0D } }  // dog
    };
    private Dictionary<int, Class> m_Classes = new Dictionary<int, Class>
    {
      { 0, new Class("cat", 0) }, // cat
      { 1, new Class("dog", 1) }  // dog
    };

    public override string SrcMark    { get { return "original"; } }
    public override string DataPath   { get { return RootPath+@"\data\cifar10trunc"; }}
    public override string OutputPath { get { return RootPath+@"\output\cifar10_original_trunc"; }}

    protected override BackpropAlgorithm CreateAlgorithm()
    {
      return Examples.CreateCIFAR10Trunc2ClassesDemo2_SEALED();
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

              Class cls;
              if (!m_Classes.TryGetValue(label, out cls))
              {
                file.Seek(3*32*32, SeekOrigin.Current);
                continue;
              }

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


      //var path = @"C:\Users\User\Desktop\net.mld";
      //DeepMethods.Models.ConvNet net;
      //using (var file = File.Open(path, FileMode.Open, FileAccess.Read))
      //  net = DeepMethods.Models.ConvNet.Deserialize(file);
      //var alg = new BackpropAlgorithm(m_TrainingSet, net);
      //net.IsTraining = false;
      //
      //var terrors = alg.GetErrors(m_TestingSet);
      //var tec = terrors.Count();
      //var tdc = m_TestingSet.Count;
      //var tpct = Math.Round(100.0F * tec / tdc, 2);
      //Console.WriteLine("Test: {0} of {1} ({2}%)", tec, tdc, tpct);
      //
      //var vcnt = m_TrainingSet.Count / 20;
      //m_ValidationSet = m_TrainingSet.Subset(0, vcnt);
      //var verrors = alg.GetErrors(m_ValidationSet);
      //var vec = verrors.Count();
      //var vdc = m_ValidationSet.Count;
      //var vpct = Math.Round(100.0F * vec / vdc, 2);
      //Console.WriteLine("Train: {0} of {1} ({2}%)", vec, vdc, vpct);
      //
      //Console.ReadLine();
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

            double[] mark;
            if (!m_Marks.TryGetValue(label, out mark))
            {
              file.Seek(3*32*32, SeekOrigin.Current);
              continue;
            }

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

            sample.Add(data, mark);
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

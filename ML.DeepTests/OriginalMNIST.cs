using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Text;
using ML.Core;
using ML.Utils;
using ML.DeepMethods.Algorithms;
using ML.DeepMethods.Models;

namespace ML.DeepTests
{
  public static class OriginalMNIST
  {
    const string MNIST_PATH       = @"C:\Users\User\Desktop\science\Machine learning\mnist";
    const string MNIST_SRC        = MNIST_PATH+@"\src\original";
    const string MNIST_TEST       = MNIST_PATH+@"\test\original";
    const string MNIST_TRAIN      = MNIST_PATH+@"\train\original";
    const string MNIST_IMG_FILE   = "img_{0}.png";
    const string MNIST_LABEL_FILE = "labels.csv";
    const string RESULTS_FOLDER   = @"..\..\..\results";

    static readonly ClassifiedSample<double[,,]> m_Training = new ClassifiedSample<double[,,]>();
    static readonly ClassifiedSample<double[,,]> m_Test     = new ClassifiedSample<double[,,]>();
    static readonly Dictionary<int, Class> m_Classes = new Dictionary<int, Class>
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

    public static void Run(string[] args)
    {
      //exportMnistData();

      loadMnistData();
      doTrain();
    }

    #region Learning

    private static void doTrain()
    {
      // create CNN
      var lenet1 = NetworkFactory.CreateLeNet1Network();
      //lenet1[lenet1.LayerCount-1].ActivationFunction = Registry.ActivationFunctions.Logistic(1);
      //ConvolutionalNetwork lenet1;
      //var filePath1 = @"F:\Work\git\ML\solution\ML.DeepTests\bin\Release\results\cnn-lenet1_1\cn_e50-0321-123745.mld";
      //using (var stream = File.Open(filePath1, FileMode.Open))
      //{
      //  lenet1 = ConvolutionalNetwork.Deserialize(stream);
      //}

      // create algorithm
      var epochs = 30;
      var alg = new BackpropAlgorithm(m_Training, lenet1)
      {
        LossFunction = Registry.LossFunctions.Euclidean,
        EpochCount = epochs,
        LearningRate = 0.005D
      };

      int epoch = 0;
      var errPct = 100.0D;
      alg.EpochEndedEvent += (o, e) =>
                             {
                               Console.WriteLine("---------------- Epoch #: {0} ({1})", ++epoch, DateTime.Now);
                               Console.WriteLine("E:\t{0}",  alg.ErrorValue);
                               Console.WriteLine("DW:\t{0}", alg.Step2);
                               Console.WriteLine("Errors:");

                               var errors = alg.GetErrors(m_Test);
                               var ec = errors.Count();
                               var dc = m_Test.Count;
                               var pct = Math.Round(100.0F * ec / dc, 2);
                               Console.WriteLine("{0} of {1} ({2}%)", ec, dc, pct);

                               if (pct < errPct)
                               {
                                 errPct = pct;
                                 var opath = Path.GetDirectoryName(System.Reflection.Assembly.GetEntryAssembly().Location)+RESULTS_FOLDER;
                                 if (!Directory.Exists(opath)) Directory.CreateDirectory(opath);
                                 var ofileName = string.Format("cn_e{0}_p{1}.mld", epoch, Math.Round(errPct, 2), DateTime.Now);
                                 var ofilePath = Path.Combine(opath, ofileName);
                                 using (var stream = File.Open(ofilePath, FileMode.Create))
                                 {
                                   lenet1.Serialize(stream);
                                 }
                               }
                               else
                               {
                                 alg.LearningRate /= 2;
                               }
                             };

      // run training process
      var now = DateTime.Now;
      Console.WriteLine();
      Console.WriteLine("Training started at {0}", now);
      alg.Train();

      Console.WriteLine("--------- ELAPSED TRAIN ----------" + (DateTime.Now-now).TotalMilliseconds);
    }

    #endregion

    #region Load

    private static void loadMnistData()
    {
      // train
      var objFilePath   = Path.Combine(MNIST_SRC, "train-images.idx3-ubyte");
      var labelFilePath = Path.Combine(MNIST_SRC, "train-labels.idx1-ubyte");
      loadSample(objFilePath, labelFilePath, m_Training);

      // test
      objFilePath   = Path.Combine(MNIST_SRC, "test-images.idx3-ubyte");
      labelFilePath = Path.Combine(MNIST_SRC, "test-labels.idx1-ubyte");
      loadSample(objFilePath, labelFilePath, m_Test);
    }

    private static void loadSample(string ipath, string lpath, ClassifiedSample<double[,,]> sample)
    {
      using (var ifile = File.Open(ipath, FileMode.Open, FileAccess.Read))
      using (var lfile = File.Open(lpath, FileMode.Open, FileAccess.Read))
      {
        var header = ReadInt32BigEndian(ifile);
        if (header != 2051) throw new Exception("Incorrect MNIST image datafile");
        header = ReadInt32BigEndian(lfile);
        if (header != 2049) throw new Exception("Incorrect MNIST label datafile");

        var count = ReadInt32BigEndian(ifile);
        var rows  = ReadInt32BigEndian(ifile);
        var cols  = ReadInt32BigEndian(ifile);

        ReadInt32BigEndian(lfile);

        for (int q=0; q<count; q++)
        {
          var data = new double[1, rows, cols];
          for (int i=0; i<rows; i++)
          for (int j=0; j<cols; j++)
          {
            var shade = ifile.ReadByte(); // do not invert 255-* because we want to keep logical format: 0=white, 255=black - not image color format!
            data[0, i, j] = shade/255.0D;
          }

          var label = lfile.ReadByte();
          sample.Add(data, m_Classes[label]);
        }

        Console.WriteLine("Loaded: {0}", ipath);
        Console.WriteLine("Loaded: {0}", lpath);
      }
    }

    #endregion

    #region Export

    private static void exportMnistData()
    {
      // train
      var objFilePath   = Path.Combine(MNIST_SRC, "train-images.idx3-ubyte");
      var labelFilePath = Path.Combine(MNIST_SRC, "train-labels.idx1-ubyte");
      exportObjects(objFilePath,  MNIST_TRAIN);
      exportLabels(labelFilePath, MNIST_TRAIN);

      // test
      objFilePath   = Path.Combine(MNIST_SRC, "test-images.idx3-ubyte");
      labelFilePath = Path.Combine(MNIST_SRC, "test-labels.idx1-ubyte");
      exportObjects(objFilePath,  MNIST_TEST);
      exportLabels(labelFilePath, MNIST_TEST);
    }

    private static void exportObjects(string fpath, string opath)
    {
      using (var file = File.Open(fpath, FileMode.Open, FileAccess.Read))
      {
        var header = ReadInt32BigEndian(file);
        if (header != 2051) throw new Exception("Incorrect MNIST image datafile");

        var count = ReadInt32BigEndian(file);
        var rows = ReadInt32BigEndian(file);
        var cols = ReadInt32BigEndian(file);

        for (int q=0; q<count; q++)
        {
          if (q%100 == 0) Console.WriteLine("Exported: {0}", q);
          exportImageData(file, opath, rows, cols, q+1);
        }
      }
    }

    private static void exportImageData(Stream ifile, string opath, int rows, int cols, int counter)
    {
      if (!Directory.Exists(opath)) Directory.CreateDirectory(opath);
      var oname = Path.Combine(opath, string.Format(MNIST_IMG_FILE, counter));

      using (var ofile = File.Open(oname, FileMode.Create, FileAccess.Write))
      {
        var image = new Bitmap(cols, rows);

        for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
        {
          var shade = 255-ifile.ReadByte(); // 0 in file means white, 255 - black
                                            // invert 255-* to map to image color
          image.SetPixel(j, i, Color.FromArgb(shade, shade, shade));
        }

        image.Save(ofile, ImageFormat.Png);

        ofile.Flush();
      }
    }

    private static void exportLabels(string fname, string opath)
    {
      var fpath = Path.Combine(MNIST_PATH, fname);

      using (var ifile = File.Open(fpath, FileMode.Open, FileAccess.Read))
      {
        var header = ReadInt32BigEndian(ifile);
        if (header != 2049) throw new Exception("Incorrect MNIST label datafile");

        if (!Directory.Exists(opath)) Directory.CreateDirectory(opath);
        var oname = Path.Combine(opath, MNIST_LABEL_FILE);
        using (var ofile = File.Open(oname, FileMode.Create, FileAccess.Write))
        using (var writer = new StreamWriter(ofile))
        {
          var count = ReadInt32BigEndian(ifile);
          for (int q=0; q<count; q++)
          {
            var label = ifile.ReadByte();
            writer.WriteLine("{0},{1}", q+1, label);
          }

          writer.Flush();
        }

        Console.WriteLine("Labels exported");
      }
    }

    #endregion

    private static int ReadInt32BigEndian(Stream stream)
    {
      var buf = new byte[4];
      stream.Read(buf, 0, 4);

      return (buf[0]<<24) | (buf[1]<<16) | (buf[2]<<8) | buf[3];
    }
  }
}

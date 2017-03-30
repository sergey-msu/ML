using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using ML.Core;
using ML.Utils;
using ML.DeepMethods.Algorithms;
using ML.DeepMethods.Models;

namespace ML.DeepTests
{
  public static class KaggleMNIST
  {
    const int IMG_SIZE = 28;

    const string MNIST_PATH       = @"C:\Users\User\Desktop\science\Machine learning\mnist";
    const string MNIST_SRC        = MNIST_PATH+@"\src\kaggle";
    const string MNIST_TEST       = MNIST_PATH+@"\test\kaggle";
    const string MNIST_TRAIN      = MNIST_PATH+@"\train\kaggle";
    const string MNIST_IMG_FILE   = "kaggle_img_{0}.png";
    const string RESULTS_FOLDER   = @"..\..\..\results";
    const string KAGGLE_RESULTS_FOLDER   = RESULTS_FOLDER+@"\mnist_kaggle";

    static readonly ClassifiedSample<double[,,]> m_Training = new ClassifiedSample<double[,,]>();
    static readonly List<double[,,]> m_Test = new List<double[,,]>();
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
      //doTrain();
      testMnistData();
    }

    #region Learning

    private static void doTrain()
    {
      // create CNN
      var lenet1 = NetworkFactory.CreateLeNet1Network();
      lenet1[lenet1.LayerCount-1].ActivationFunction = Registry.ActivationFunctions.Logistic(1);
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
        LossFunction = Registry.LossFunctions.CrossEntropySoftMax,
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

                               var errors = alg.GetErrors(m_Training); // we do not have public test data in kaggle :(
                               var ec = errors.Count();
                               var dc = m_Training.Count;
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

    private static void testMnistData()
    {
      ConvolutionalNetwork lenet1;
      var opath = Path.GetDirectoryName(System.Reflection.Assembly.GetEntryAssembly().Location)+KAGGLE_RESULTS_FOLDER;
      var fpath = Path.Combine(opath, "cn_e26_p0.06.mld");
      //var fpath = @"F:\Work\git\ML\solution\ML.DigitsDemo\lenet1.mld";

      using (var stream = File.Open(fpath, FileMode.Open))
      {
        lenet1 = ConvolutionalNetwork.Deserialize(stream);
      }
      var alg = new BackpropAlgorithm(m_Training, lenet1);

      var fout = Path.Combine(MNIST_SRC, "result1.csv");
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

    #region Load

    private static void loadMnistData()
    {
      // train
      var objFilePath = Path.Combine(MNIST_SRC, "train.csv");
      loadSample(objFilePath, m_Training);

      // test
      objFilePath = Path.Combine(MNIST_SRC, "test.csv");
      loadTest(objFilePath, m_Test);
    }

    private static void loadSample(string ipath, ClassifiedSample<double[,,]> sample)
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

          var data = new double[1, IMG_SIZE, IMG_SIZE];
          for (int i=1; i<=IMG_SIZE*IMG_SIZE; i++)
          {
            var shade = raw[i]; // do not invert 255-* because we want to keep logical format: 0=white, 255=black - not image color format!
            var x = (i-1)%IMG_SIZE;
            var y = (i-1)/IMG_SIZE;
            data[0, y, x] = shade/255.0D;
          }
          sample.Add(data, m_Classes[label]);
        }

        Console.WriteLine("Loaded: {0}", ipath);
      }
    }

    private static void loadTest(string ipath, List<double[,,]> sample)
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

          var data = new double[1, IMG_SIZE, IMG_SIZE];
          for (int i=0; i<IMG_SIZE*IMG_SIZE; i++)
          {
            var shade = raw[i]; // do not invert 255-* because we want to keep logical format: 0=white, 255=black - not image color format!
            var x = i%IMG_SIZE;
            var y = i/IMG_SIZE;
            data[0, y, x] = shade/255.0D;
          }
          sample.Add(data);
        }

        Console.WriteLine("Loaded: {0}", ipath);
      }
    }

    #endregion

    #region Export

    private static void exportMnistData()
    {
      // train
      var objFilePath = Path.Combine(MNIST_SRC, "train.csv");
      exportObjects(objFilePath, MNIST_TRAIN, true);

      // test
      objFilePath = Path.Combine(MNIST_SRC, "test.csv");
      exportObjects(objFilePath, MNIST_TEST, false);
    }

    private static void exportObjects(string fpath, string opath, bool isTrain)
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

    private static void exportImageData(int[] data, string opath, int counter)
    {
      if (!Directory.Exists(opath)) Directory.CreateDirectory(opath);
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
  }
}

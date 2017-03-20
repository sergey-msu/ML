﻿using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Text;
using ML.Core;
using ML.Utils;
using ML.DeepMethods.Algorithms;

namespace ML.DeepTests
{
  class Program
  {
    const string MNIST_PATH       = @"C:\Users\User\Desktop\science\Machine learning\mnist";
    const string MNIST_SRC        = MNIST_PATH+@"\src";
    const string MNIST_TEST       = MNIST_PATH+@"\test";
    const string MNIST_TRAIN      = MNIST_PATH+@"\train";
    const string MNIST_IMG_FILE   = "img_{0}.png";
    const string MNIST_LABEL_FILE = "labels.csv";

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

    static void Main(string[] args)
    {
      //exportMnistData();

      loadMnistData();
      doTrain();

      Console.ReadLine();
    }

    #region Learning

    static void doTrain()
    {
      // create CNN
      var lenet1 = NetworkFactory.CreateLeNet1Network();

      // create algorithm
      var alg = new BackpropAlgorithm(m_Training, lenet1)
      {
        EpochCount = 6000,
        LearningRate = 0.1D
      };
      int epoch = 0;
      alg.EpochEndedEvent += (o, e) =>
                             {
                               //if (epoch++ % 300 != 0) return;
                               Console.WriteLine("----------------Epoch #: {0}", epoch);
                               Console.WriteLine("E:\t{0}",  alg.ErrorValue);
                               Console.WriteLine("DE:\t{0}", alg.ErrorDelta);
                               Console.WriteLine("Q:\t{0}",  alg.QValue);
                               Console.WriteLine("DQ:\t{0}", alg.QDelta);
                               Console.WriteLine("DW:\t{0}", alg.Step2);
                             };

      // run training process
      var now = DateTime.Now;
      Console.WriteLine();
      Console.WriteLine("Training started at {0}", now);
      alg.Train();

      // output results

      Console.WriteLine("--------- ELAPSED TRAIN ----------" + (DateTime.Now-now).TotalMilliseconds);
      Console.WriteLine("Error function: " + alg.ErrorValue);
      Console.WriteLine("Step: " + alg.Step2);
      Console.WriteLine("Errors:");

      var errors = alg.GetErrors(m_Test);
      var ec = errors.Count();
      var dc = m_Test.Count;
      var pct = Math.Round(100.0F * ec / dc, 2);
      Console.WriteLine("{0} of {1} ({2}%)", ec, dc, pct);
    }


    #endregion

    #region Load

    static void loadMnistData()
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

    static void loadSample(string ipath, string lpath, ClassifiedSample<double[,,]> sample)
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
          if (q%10000 == 0) Console.WriteLine("Loaded: {0}", q);

          var data = new double[1, rows, cols];
          for (int i=0; i<rows; i++)
          for (int j=0; j<cols; j++)
          {
            var value = ifile.ReadByte();
            var shade = 255-value;
            data[0, i, j] = shade/255.0D;
          }

          var label = lfile.ReadByte();
          sample.Add(data, m_Classes[label]);
        }
      }
    }

    #endregion

    #region Export

    static void exportMnistData()
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

    static void exportObjects(string fpath, string opath)
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
          readImageData(file, opath, rows, cols, q+1);
        }
      }
    }

    static void readImageData(Stream ifile, string opath, int rows, int cols, int counter)
    {
      if (!Directory.Exists(opath)) Directory.CreateDirectory(opath);
      var oname = Path.Combine(opath, string.Format(MNIST_IMG_FILE, counter));

      using (var ofile = File.Open(oname, FileMode.OpenOrCreate, FileAccess.Write))
      {
        var image = new Bitmap(cols, rows);

        for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
        {
          var value = ifile.ReadByte();
          var shade = 255-value;
          image.SetPixel(j, i, Color.FromArgb(shade, shade, shade));
        }

        image.Save(ofile, ImageFormat.Png);

        ofile.Flush();
      }
    }

    static void exportLabels(string fname, string opath)
    {
      var fpath = Path.Combine(MNIST_PATH, fname);

      using (var ifile = File.Open(fpath, FileMode.Open, FileAccess.Read))
      {
        var header = ReadInt32BigEndian(ifile);
        if (header != 2049) throw new Exception("Incorrect MNIST label datafile");

        if (!Directory.Exists(opath)) Directory.CreateDirectory(opath);
        var oname = Path.Combine(opath, MNIST_LABEL_FILE);
        using (var ofile = File.Open(oname, FileMode.OpenOrCreate, FileAccess.Write))
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

    static int ReadInt32BigEndian(Stream stream)
    {
      var buf = new byte[4];
      stream.Read(buf, 0, 4);

      return (buf[0]<<24) | (buf[1]<<16) | (buf[2]<<8) | buf[3];
    }
  }
}

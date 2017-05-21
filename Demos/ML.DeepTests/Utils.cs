using System;
using System.IO;
using System.Linq;
using ML.Contracts;
using ML.Core;
using ML.DeepMethods.Algorithms;

namespace ML.DeepTests
{
  public static class Utils
  {
    public static void HandleEpochEnded(BackpropAlgorithm alg, MultiRegressionSample<double[][,]> test, MultiRegressionSample<double[][,]> train, Class[] classes, string outputPath)
    {
      Console.WriteLine("---------------- Epoch #{0} ({1})", alg.Epoch, DateTime.Now);
      Console.WriteLine("L:\t{0}", alg.LossValue);
      Console.WriteLine("DW:\t{0}", alg.Step2);
      Console.WriteLine("LR:\t{0}", alg.LearningRate);

      double? pct = null;

      if (test==null || !test.Any())
        Console.WriteLine("Test: none");
      else
      {
        var terrors = alg.GetClassificationErrors(test, classes);
        var tec = terrors.Count();
        var tdc = test.Count;
        var tpct = Math.Round(100.0F * tec / tdc, 2);
        Console.WriteLine("Test: {0} of {1} ({2}%)", tec, tdc, tpct);

        pct = tpct;
      }

      if (train==null || !train.Any())
        Console.WriteLine("Train: none");
      else
      {
        var verrors = alg.GetClassificationErrors(train, classes);
        var vec = verrors.Count();
        var vdc = train.Count;
        var vpct = Math.Round(100.0F * vec / vdc, 2);
        Console.WriteLine("Train: {0} of {1} ({2}%)", vec, vdc, vpct);

        if (!pct.HasValue) pct=vpct;
      }

      var ofileName = string.Format("cn_e{0}_p{1}.mld", alg.Epoch, Math.Round(pct.Value, 2));
      var ofilePath = Path.Combine(outputPath, ofileName);
      using (var stream = File.Open(ofilePath, FileMode.Create))
      {
        alg.Net.Serialize(stream);
      }
    }

    public static void HandleBatchEnded(BackpropAlgorithm alg, int trainCount, DateTime tstart)
    {
      var now = DateTime.Now;
      var iter = alg.Iteration;
      var pct = 100*iter/(float)trainCount;
      var elapsed = TimeSpan.FromMinutes((now-tstart).TotalMinutes * (trainCount-iter)/iter);
      Console.Write("\rCurrent epoch progress: {0:0.00}%. Left {1:00}m {2:00}s.  L={3:0.0000}         ",
                    pct,
                    elapsed.Minutes,
                    elapsed.Seconds,
                    alg.LossValue);
    }

    public static void SaveAlgCrushResults(BackpropAlgorithm alg, string outputPath)
    {
      var ofileName = string.Format("cn_e{0}_crush.mld", alg.Epoch);
      var ofilePath = Path.Combine(outputPath, ofileName);
      using (var stream = File.Open(ofilePath, FileMode.Create))
      {
        alg.Net.Serialize(stream);
      }
    }


    public static void Convolve(string srcPath, string kernelPath, string outPath)
    {
      var data = new double[1][,] { new double[32,32] };
      var ker = new double[1][,] { new double[9,9] };

      using (var cat    = new System.Drawing.Bitmap(srcPath))
      using (var kernel = new System.Drawing.Bitmap(kernelPath))
      {
        for (int x=0; x<32; x++)
        for (int y=0; y<32; y++)
          data[0][y, x] = (255.0D-cat.GetPixel(x, y).B)/255.0D;

        for (int x=0; x<9; x++)
        for (int y=0; y<9; y++)
          ker[0][y, x] = (255.0D-kernel.GetPixel(x, y).B)/255.0D;
      }

      var convolution = new DeepMethods.Models.ConvLayer(1, 9, padding: 4);
      convolution.InputDepth = 1;
      convolution.InputHeight = 32;
      convolution.InputWidth = 32;
      convolution._Build();

      for (int y=0; y<9; y++)
      for (int x=0; x<9; x++)
        convolution.SetKernel(0, 0, y, x, ker[0][y, x]);

      var res = convolution.Calculate(data);
      var max = res[0].Cast<double>().Max();

      using (var fm = new System.Drawing.Bitmap(32, 32))
      {
        for (int x=0; x<32; x++)
        for (int y=0; y<32; y++)
        {
          var val = 255.0D*(1.0D - res[0][y, x]/max);
          if (val>128) val = 255;
          fm.SetPixel(x, y, System.Drawing.Color.FromArgb((int)val, (int)val, (int)val));
        }

        fm.Save(outPath);
      }
    }

  }
}

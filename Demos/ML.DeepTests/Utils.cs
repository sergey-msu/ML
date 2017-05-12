using System;
using System.IO;
using System.Linq;
using ML.Core;
using ML.DeepMethods.Algorithms;

namespace ML.DeepTests
{
  public static class Utils
  {
    public static void HandleEpochEnded(BackpropAlgorithm alg, ClassifiedSample<double[][,]> test, ClassifiedSample<double[][,]> validation, string outputPath)
    {
      Console.WriteLine("---------------- Epoch #: {0} ({1})", alg.Epoch, DateTime.Now);
      Console.WriteLine("L:\t{0}", alg.LossValue);
      Console.WriteLine("DW:\t{0}", alg.Step2);
      Console.WriteLine("LR:\t{0}", alg.LearningRate);

      var terrors = alg.GetErrors(test);
      var tec = terrors.Count();
      var tdc = test.Count;
      var tpct = Math.Round(100.0F * tec / tdc, 2);
      Console.WriteLine("Test: {0} of {1} ({2}%)", tec, tdc, tpct);

      var verrors = alg.GetErrors(validation);
      var vec = verrors.Count();
      var vdc = validation.Count;
      var vpct = Math.Round(100.0F * vec / vdc, 2);
      Console.WriteLine("Validation: {0} of {1} ({2}%)", vec, vdc, vpct);

      var ofileName = string.Format("cn_e{0}_p{1}.mld", alg.Epoch, Math.Round(tpct, 2));
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

  }
}

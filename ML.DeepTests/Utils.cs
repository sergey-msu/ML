using System;
using System.IO;
using System.Linq;
using ML.Core;
using ML.DeepMethods.Algorithms;

namespace ML.DeepTests
{
  public static class Utils
  {
    public static void HandleEpochEnded(BackpropAlgorithm alg, ClassifiedSample<double[][,]> test, string outputPath)
    {
      Console.WriteLine("---------------- Epoch #: {0} ({1})", alg.Epoch, DateTime.Now);
      Console.WriteLine("L:\t{0}", alg.LossValue);
      Console.WriteLine("DW:\t{0}", alg.Step2);
      Console.WriteLine("LR:\t{0}", alg.LearningRate);
      Console.WriteLine("Errors:");

      var errors = alg.GetErrors(test);
      var ec = errors.Count();
      var dc = test.Count;
      var pct = Math.Round(100.0F * ec / dc, 2);
      Console.WriteLine("{0} of {1} ({2}%)", ec, dc, pct);

      var ofileName = string.Format("cn_e{0}_p{1}.mld", alg.Epoch, Math.Round(pct, 2));
      var ofilePath = Path.Combine(outputPath, ofileName);
      using (var stream = File.Open(ofilePath, FileMode.Create))
      {
        alg.Result.Serialize(stream);
      }
    }

    public static void SaveAlgCrushResults(BackpropAlgorithm alg, string outputPath)
    {
      var ofileName = string.Format("cn_e{0}_crush.mld", alg.Epoch);
      var ofilePath = Path.Combine(outputPath, ofileName);
      using (var stream = File.Open(ofilePath, FileMode.Create))
      {
        alg.Result.Serialize(stream);
      }
    }

  }
}

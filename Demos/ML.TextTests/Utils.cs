using System;
using System.IO;
using System.Linq;
using ML.Contracts;
using ML.TextMethods.Algorithms;
using ML.Core;

namespace ML.TextTests
{
  public static class Utils
  {
    public static void HandleTrainEnded(TextAlgorithmBase alg,
                                        ClassifiedSample<string> test,
                                        string outputPath)
    {
      Console.WriteLine("\r------------------------------------------- {0}: Training finished at {1}", alg.Name, DateTime.Now);

      var terrors = alg.GetErrors(test, 0, true);
      var tec = terrors.Count();
      var tdc = test.Count;
      var pct = Math.Round(100.0F * tec / tdc, 2);

      Console.WriteLine("Test: {0} of {1} ({2}%)", tec, tdc, pct);

      var ofileName = string.Format("cn_{0}_p{1}.mld", alg.Name, Math.Round(pct, 2));
      var ofilePath = Path.Combine(outputPath, ofileName);
      using (var stream = File.Open(ofilePath, FileMode.Create))
      {
        alg.Serialize(stream);
      }
    }

    public static void SaveAlgCrushResults(TextAlgorithmBase alg, string outputPath)
    {
      var ofileName = string.Format("cn_e{0}_{1}_crush.mld", alg.Name, (DateTime.Now-DateTime.MinValue).TotalMilliseconds);
      var ofilePath = Path.Combine(outputPath, ofileName);
      using (var stream = File.Open(ofilePath, FileMode.Create))
      {
        alg.Serialize(stream);
      }
    }
  }
}

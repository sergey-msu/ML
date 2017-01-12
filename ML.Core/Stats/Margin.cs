using System;
using System.Collections.Generic;
using System.Linq;
using ML.Core.Contracts;

namespace ML.Core.Stats
{
  public static class Margin
  {
    public static Dictionary<int, float> Calculate(IAlgorithm algorithm)
    {
      var result = new SortedDictionary<int, float>();
      int idx = -1;

      foreach (var pData in algorithm.TrainingSample)
      {
        idx++;
        float maxi = float.MinValue;
        float si = 0;

        foreach (var cls in algorithm.Classes)
        {
          var closeness = algorithm.EstimateClose(pData.Key, cls.Value);
          if (cls.Value == pData.Value) si = closeness;
          else
          {
            if (maxi < closeness) maxi = closeness;
          }
        }

        result.Add(idx, si - maxi);
      }

      return result.OrderBy(r => r.Value).ToDictionary(r => r.Key, r => r.Value);
    }

  }
}

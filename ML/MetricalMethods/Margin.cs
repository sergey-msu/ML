using System;
using System.Collections.Generic;
using System.Linq;
using ML.Contracts;

namespace ML.MetricalMethods
{
  /// <summary>
  /// Margin calculator
  /// </summary>
  public static class Margin
  {
    /// <summary>
    /// Calculates margin for given algorithm
    /// </summary>
    public static Dictionary<int, float> Calculate(IMetricAlgorithm algorithm)
    {
      var result = new SortedDictionary<int, float>();
      int idx = -1;

      foreach (var pData in algorithm.TrainingSample)
      {
        idx++;
        float maxi = float.MinValue;
        float si = 0;

        foreach (var cls in algorithm.Classes.Values)
        {
          var closeness = algorithm.EstimateClose(pData.Key, cls);
          if (cls == pData.Value) si = closeness;
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

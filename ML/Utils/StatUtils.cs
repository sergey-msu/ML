using System;
using System.Collections.Generic;
using System.Linq;
using ML.Contracts;
using ML.MetricMethods.Algorithms;

namespace ML.Utils
{
  public static class StatUtils
  {
    /// <summary>
    /// Calculates margins
    /// </summary>
    public static Dictionary<int, double> CalculateMargins<TObj>(IGammaAlgorithm<TObj> alg)
    {
      var result = new Dictionary<int, double>();
      int idx = -1;

      foreach (var pData in alg.TrainingSample)
      {
        idx++;
        double maxi = double.MinValue;
        double si = 0;

        foreach (var cls in alg.TrainingSample.Classes)
        {
          var proximity = alg.CalculateClassScore(pData.Key, cls);
          if (cls==pData.Value) si = proximity;
          else
          {
            if (maxi < proximity) maxi = proximity;
          }
        }

        result.Add(idx, si - maxi);
      }

      return result.OrderBy(r => r.Value).ToDictionary(r => r.Key, r => r.Value);
    }


    /// <summary>
    /// Leave-one-out optimization
    /// </summary>
    public static void OptimizeLOO<TObj>(IKernelAlgorithm<TObj> alg, double hMin, double hMax, double step)
    {
      var hOpt = double.MaxValue;
      var minErrCnt = int.MaxValue;

      for (double h = hMin; h <= hMax; h += step)
      {
        var errCnt = 0;
        alg.H = h;

        var initSample = alg.TrainingSample;

        for (int i=0; i<initSample.Count; i++)
        {
          var pData = initSample.ElementAt(i);
          var looSample  = initSample.ApplyMask((p, c, idx) => idx != i);

          try
          {
            alg.Train(looSample);

            var predClass = alg.Predict(pData.Key);
            var realClass = pData.Value;
            if (predClass != realClass) errCnt++;
          }
          finally
          {
            alg.Train(initSample);
          }
        }

        if (errCnt < minErrCnt)
        {
          minErrCnt = errCnt;
          hOpt = h;
        }
      }

      alg.H = hOpt;
    }

    /// <summary>
    /// Leave-one-out optimization
    /// </summary>
    public static void OptimizeLOO(NearestKNeighboursAlgorithm alg, int? minK = null, int? maxK = null)
    {
      if (!minK.HasValue || minK.Value<1) minK = 1;
      if (!maxK.HasValue || maxK.Value>alg.TrainingSample.Count) maxK = alg.TrainingSample.Count-1;

      var kOpt = int.MaxValue;
      var minErrCnt = int.MaxValue;

      for (int k=minK.Value; k<=maxK.Value; k++)
      {
        var errCnt = 0;
        alg.K = k;

        var initSample = alg.TrainingSample;

        foreach (var pData in initSample)
        {
          var looSample  = initSample.ApplyMask((p, c, idx) => p != pData.Key);

          try
          {
            alg.TrainingSample = looSample;

            var predClass = alg.Predict(pData.Key);
            var realClass = pData.Value;
            if (predClass != realClass) errCnt++;
          }
          finally
          {
            alg.TrainingSample = initSample;
          }
        }

        if (errCnt < minErrCnt)
        {
          minErrCnt = errCnt;
          kOpt = k;
        }
      }

      alg.K = kOpt;
    }

  }
}

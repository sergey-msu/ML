using System;
using System.Collections.Generic;
using System.Linq;
using ML.Contracts;
using ML.Core;

namespace ML.MetricMethods.Algorithms
{
  /// <summary>
  /// Nearest K Neighbours Algorithm
  /// </summary>
  public sealed class NearestKNeighboursAlgorithm : OrderedMetricAlgorithmBase
  {
    private int m_K;

    public NearestKNeighboursAlgorithm(IMetric metric, int k)
      : base(metric)
    {
      K = k;
    }

    /// <summary>
    /// Algorithm mnemonic ID
    /// </summary>
    public override string ID { get { return "KNN"; } }

    /// <summary>
    /// Algorithm name
    /// </summary>
    public override string Name { get { return "K-Nearest Neighbor(s)"; } }

    /// <summary>
    /// Neighbour count
    /// </summary>
    public int K
    {
      get { return m_K; }
      set
      {
        if (value <= 0)
          throw new MLException("NearestKNeighboursAlgorithm.K(value<=0)");

        m_K = value;
      }
    }


    /// <summary>
    /// Calculate 'weight' - a contribution of training point (i-th from ordered training sample)
    /// to closeness of test point to its class
    /// </summary>
    /// <param name="i">Point number in ordered training sample</param>
    /// <param name="x">Test point</param>
    /// <param name="orderedSample">Ordered training sample</param>
    protected override double CalculateWeight(int i, double[] x, Dictionary<double[], double> orderedSample)
    {
      return (i < m_K) ? 1 : 0;
    }

    /// <summary>
    /// Leave-one-out optimization
    /// </summary>
    public void OptimizeLOO(int? minK = null, int? maxK = null)
    {
      if (!minK.HasValue || minK.Value<1) minK = 1;
      if (!maxK.HasValue || maxK.Value>TrainingSample.Count) maxK = TrainingSample.Count;

      var kOpt = int.MaxValue;
      var minErrCnt = int.MaxValue;

      for (int k=minK.Value; k<=maxK.Value; k++)
      {
        var errCnt = 0;
        m_K = k;

        var initSample = TrainingSample;

        for (int i=0; i<initSample.Count; i++)
        {
          var pData = initSample.ElementAt(i);
          var looSample  = initSample.ApplyMask((p, c, idx) => idx != i);
          TrainingSample = looSample;

          var predClass = this.Predict(pData.Key);
          var realClass = pData.Value;
          if (predClass != realClass) errCnt++;

          TrainingSample = initSample;
        }

        if (errCnt < minErrCnt)
        {
          minErrCnt = errCnt;
          kOpt = k;
        }
      }

      m_K = kOpt;
    }
  }
}

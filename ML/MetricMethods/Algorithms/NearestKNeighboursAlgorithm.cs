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
  public sealed class NearestKNeighboursAlgorithm : OrderedMetricAlgorithmBase<double[]>
  {
    private int m_K;

    public NearestKNeighboursAlgorithm(IMetric<double[]> metric, int k)
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
    /// Classify point
    /// </summary>
    public override Class Predict(double[] obj)
    {
      var scores = new List<KeyValuePair<Class, double>>();
      var len = 0;

      foreach (var pData in TrainingSample)
      {
        var dist = Metric.Dist(obj, pData.Key);

        if (len<m_K)
        {
          scores.Add(new KeyValuePair<Class, double>(pData.Value, dist));
          len++;
          continue;
        }

        var maxDist = -1.0D;
        int idx = -1;
        for (int i=0; i<m_K; i++)
        {
          var mData = scores[i];
          var mDist = mData.Value;
          if (mDist>maxDist)
          {
            maxDist = mDist;
            idx = i;
          }
        }

        if (dist<maxDist)
          scores[idx] = new KeyValuePair<Class, double>(pData.Value, dist);
      }

      var hist = new Dictionary<Class, int>();
      for (int i=0; i<m_K; i++)
      {
        var mData = scores[i];

        if (!hist.ContainsKey(mData.Key)) hist[mData.Key] = 1;
        else hist[mData.Key]++;
      }

      var max = double.MinValue;
      Class result = null;
      foreach (var h in hist)
      {
        if (h.Value>max)
        {
          max = h.Value;
          result = h.Key;
        }
      }

      return result;
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
  }
}

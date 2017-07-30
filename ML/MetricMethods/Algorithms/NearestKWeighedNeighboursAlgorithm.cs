using System;
using System.Linq;
using System.Collections.Generic;
using ML.Contracts;
using ML.Core;

namespace ML.MetricMethods.Algorithms
{
  /// <summary>
  /// Nearest K Neighbours with Weights Algorithm
  /// </summary>
  public sealed class NearestKWeighedNeighboursAlgorithm : OrderedMetricAlgorithmBase<double[]>
  {
    private int m_K;
    private double[] m_Weights;

    public NearestKWeighedNeighboursAlgorithm(IMetric<double[]> metric,
                                              int k,
                                              double[] weights)
      : base(metric)
    {
      K = k;
      Weights = weights;
    }

    /// <summary>
    /// Algorithm mnemonic ID
    /// </summary>
    public override string Name { get { return "WNNK"; } }

    /// <summary>
    /// Neighbour count
    /// </summary>
    public int K
    {
      get { return m_K; }
      set
      {
        if (value <= 0)
          throw new MLException("NearestKWeighedNeighboursAlgorithm.K(value<=0)");

        m_K = value;
      }
    }

    /// <summary>
    /// Neighbout weights
    /// </summary>
    public double[] Weights
    {
      get { return m_Weights; }
      set
      {
        double[] weights;
        if (value==null)
        {
          weights = new double[m_K];
          for (int i=0; i<m_K; i++)
            weights[i] = 1;
        }
        else
        {
          weights = value.ToArray();
        }

        m_Weights = weights;
      }
    }

    /// <summary>
    /// Calculates 'weight' - a contribution of training point (i-th from ordered training sample)
    /// to closeness of test point to its class
    /// </summary>
    /// <param name="i">Point number in ordered training sample</param>
    /// <param name="x">Test point</param>
    /// <param name="orderedSample">Ordered training sample</param>
    protected override double CalculateWeight(int i, double[] x, Dictionary<double[], double> orderedSample)
    {
      return (i < m_K && i < m_Weights.Length) ? m_Weights[i] : 0;
    }
  }
}

using System;
using System.Collections.Generic;
using System.Linq;
using ML.Core;
using ML.Core.Mathematics;

namespace ML.NeuralMethods.Algorithms
{
  /// <summary>
  /// Neural Network machine learning algorithm
  /// </summary>
  public abstract class NeuralNetworkAlgorithmBase : AlgorithmBase
  {
    protected NeuralNetworkAlgorithmBase(ClassifiedSample classifiedSample)
      : base(classifiedSample)
    {
      m_Network = new NeuralNetwork();
    }

    private readonly NeuralNetwork m_Network;

    public NeuralNetwork Network { get { return m_Network; } }


    public override Class Classify(Point x)
    {
      var result = m_Network.Calculate(x);

      int idx;
      double max;
      MathUtils.CalcMax(result, out idx, out max);

      var cls = Classes.FirstOrDefault(c => (int)c.Value.Value == idx);

      return cls.Value ?? Class.None;
    }
  }
}

using System;
using System.Collections.Generic;
using System.Linq;
using ML.Core;
using ML.Core.Mathematics;
using ML.NeuralMethods.Model;

namespace ML.NeuralMethods.Algorithms
{
  /// <summary>
  /// Feedforward Neural Network machine learning algorithm
  /// </summary>
  public abstract class NeuralNetworkAlgorithmBase : AlgorithmBase
  {
    protected NeuralNetworkAlgorithmBase(ClassifiedSample classifiedSample)
      : base(classifiedSample)
    {
    }

    private NeuralNetwork m_Result;

    /// <summary>
    /// The result of the algorithm
    /// </summary>
    public NeuralNetwork Result { get { return m_Result; } }


    /// <summary>
    /// Maps object to corresponding class
    /// </summary>
    public override Class Classify(Point x)
    {
      var result = m_Result.Calculate(x);
      var len = result.Length;
      Class cls;

      if (len==1)
      {
        cls = Classes.FirstOrDefault(c => (int)c.Value.Value == (int)result[0]).Value ?? Class.None;
      }
      else
      {
        int idx;
        double max;
        MathUtils.CalcMax(result, out idx, out max);

        cls = Classes.FirstOrDefault(c => (int)c.Value.Value == idx+1).Value  ?? Class.None;
      }

      return cls;
    }

    /// <summary>
    /// Teaches algorithm, produces Network output
    /// </summary>
    public void Train()
    {
      m_Result = DoTrain();
    }

    protected abstract NeuralNetwork DoTrain();
  }
}

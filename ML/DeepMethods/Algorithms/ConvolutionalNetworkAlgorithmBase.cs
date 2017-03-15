using System;
using System.Collections.Generic;
using System.Linq;
using ML.Core;
using ML.Core.Mathematics;
using ML.DeepMethods.Model;

namespace ML.DeepMethods.Algorithms
{
  /// <summary>
  /// Feedforward Convolutional Neural Network machine learning algorithm
  /// </summary>
  public abstract class ConvolutionalNetworkAlgorithmBase: AlgorithmBase
  {
    private ConvolutionalNetwork m_Result;

    protected ConvolutionalNetworkAlgorithmBase(ClassifiedSample trainingSample, ConvolutionalNetwork net)
      : base(trainingSample)
    {
      if (net==null)
        throw new MLException("Network can not be null");

      m_Result = net;
    }

    /// <summary>
    /// The result of the algorithm
    /// </summary>
    public ConvolutionalNetwork Result { get { return m_Result; } }

    public override Class Classify(object obj)
    {
      return Classify((double[,,])obj);
    }

    /// <summary>
    /// Maps object to corresponding class
    /// </summary>
    public Class Classify(double[,,] input)
    {
      var result = m_Result.Calculate(input);
      var len = result.GetLength(0);
      Class cls;

      if (len==1)
      {
        cls = m_Classes.FirstOrDefault(c => (int)c.Value.Value == (int)result[0,0,0]).Value ?? Class.None;
      }
      else
      {
        int iidx;
        int jidx;
        int kidx;
        double max;
        MathUtils.CalcMax(result, out iidx, out jidx, out kidx, out max);

        cls = m_Classes.FirstOrDefault(c => (int)c.Value.Value == iidx+1).Value  ?? Class.None;
      }

      return cls;
    }

    /// <summary>
    /// Teaches algorithm, produces Network output
    /// </summary>
    public abstract void Train();
  }
}

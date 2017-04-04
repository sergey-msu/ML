using System;
using System.Collections.Generic;
using System.Linq;
using ML.Core;
using ML.Core.Mathematics;
using ML.DeepMethods.Models;

namespace ML.DeepMethods.Algorithms
{
  /// <summary>
  /// Feedforward Convolutional Neural Network machine learning algorithm
  /// </summary>
  public abstract class ConvolutionalNetworkAlgorithmBase: AlgorithmBase<double[,,]>
  {
    private ConvolutionalNetwork m_Result;

    protected ConvolutionalNetworkAlgorithmBase(ClassifiedSample<double[,,]> trainingSample, ConvolutionalNetwork net)
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

    /// <summary>
    /// Maps object to corresponding class
    /// </summary>
    public override Class Classify(double[,,] input)
    {
      var result = m_Result.Calculate(input);
      var len = result.GetLength(0);
      Class cls;

      int iidx;
      int jidx;
      int kidx;
      double max;
      MathUtils.CalcMax(result, out iidx, out jidx, out kidx, out max);

      cls = m_Classes.FirstOrDefault(c => (int)c.Value.Value == iidx).Value  ?? Class.None;

      return cls;
    }

    public override IEnumerable<ErrorInfo> GetErrors(ClassifiedSample<double[,,]> classifiedSample)
    {
      var isTraining = m_Result.IsTraining;
      m_Result.IsTraining = false;
      try
      {
        return base.GetErrors(classifiedSample);
      }
      finally
      {
        m_Result.IsTraining = isTraining;
      }

    }

    /// <summary>
    /// Teaches algorithm, produces Result output
    /// </summary>
    public void Train()
    {
      m_Result.IsTraining = true;
      try
      {
        DoTrain();
      }
      finally
      {
        m_Result.IsTraining = false;
      }
    }

    protected abstract void DoTrain();
  }
}

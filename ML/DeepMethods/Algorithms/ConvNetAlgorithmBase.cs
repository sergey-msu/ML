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
  public abstract class ConvNetAlgorithmBase: AlgorithmBase<double[][,]>
  {
    private ConvNet m_Net;

    protected ConvNetAlgorithmBase(ClassifiedSample<double[][,]> trainingSample, ConvNet net)
      : base(trainingSample)
    {
      if (net==null)
        throw new MLException("Network can not be null");

      m_Net = net;
    }

    /// <summary>
    /// The result of the algorithm
    /// </summary>
    public ConvNet Net { get { return m_Net; } }

    /// <summary>
    /// Maps object to corresponding class
    /// </summary>
    public override Class Classify(double[][,] input)
    {
      var result = m_Net.Calculate(input);
      var res = MathUtils.ArgMax<double>(result);
      var cls = m_Classes.FirstOrDefault(c => (int)c.Value.Value == res).Value  ?? Class.None;

      return cls;
    }

    public override IEnumerable<ErrorInfo> GetErrors(ClassifiedSample<double[][,]> classifiedSample)
    {
      var isTraining = m_Net.IsTraining;
      m_Net.IsTraining = false;
      try
      {
        return base.GetErrors(classifiedSample);
      }
      finally
      {
        m_Net.IsTraining = isTraining;
      }
    }

    /// <summary>
    /// Teaches algorithm, produces Result output
    /// </summary>
    public void Train()
    {
      m_Net.IsTraining = true;
      try
      {
        Build();
        DoTrain();
      }
      finally
      {
        m_Net.IsTraining = false;
      }
    }

    public abstract void Build();

    protected abstract void DoTrain();
  }
}

using System;
using System.Collections.Generic;
using System.Linq;
using ML.Core;

namespace ML.NeuralMethods.Algorithms
{
  /// <summary>
  /// Neural Network machine learning algorithm
  /// </summary>
  public class NeuralNetworkAlgorithm : AlgorithmBase
  {
    #region Inner

    public class NeuronSchema : List<int> { }
    public class LayerSchema : List<NeuronSchema> { }
    public class NetSchema : List<LayerSchema> { }

    #endregion

    public NeuralNetworkAlgorithm(ClassifiedSample classifiedSample, NetSchema schema)
      : base(classifiedSample)
    {
      if (schema==null || !schema.Any())
        throw new MLException("NeuralNetworkAlgorithm.ctor(schema=null|empty)");

      m_Network = new NeuralNetwork<Point>();
    }

    private readonly NeuralNetwork<Point> m_Network;

    public override string ID { get { return "NNET"; } }
    public override string Name { get { return "Neural Network"; } }
    public NeuralNetwork<Point> Network { get { return m_Network; } }


    public override Class Classify(Point x)
    {
      var  result = m_Network.Calculate(x);
      var idx = -1;
      var max = double.MinValue;

      for (int i=0; i<result.Length; i++)
      {
        var val = result[i];
        if (idx<0 || val > max)
        {
          idx = i;
          max = val;
        }
      }

      var cls = Classes.FirstOrDefault(c => c.Value.Order == idx);

      return cls.Value ?? Class.None;
    }

    #region .pvt

    private void initNetwork(NetSchema schema)
    {
      foreach (var ls in schema)
      {
        var layer = m_Network.AddLayer();
        foreach (var ns in ls)
        {
          var neuron = layer.AddNeuron();
          foreach (var ws in ns)
            neuron[ws] = 0.0D;
        }
      }
    }

    #endregion
  }
}

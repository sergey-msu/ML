using System;
using System.Collections.Generic;
using ML.Contracts;
using ML.Core.ComputingNetworks;
using ML.Core;

namespace ML.NeuralMethods.Networks
{
  /// <summary>
  /// Represents artificial neural network: set of layers with neuron nodes and weighted connections
  /// </summary>
  public class NeuralNetwork : SequenceNode<double[], NeuralLayer>
  {
    public IFunction m_ActivationFunction;

    public NeuralNetwork()
    {
    }

    /// <summary>
    /// Dimension of input vector
    /// </summary>
    public int InputDim { get; set; }

    /// <summary>
    /// If true, adds artificial +1 input value in the and of input data array
    /// </summary>
    public bool UseBias { get; set; }

    /// <summary>
    /// Layer activation function. If null, the network's activation function will be used
    /// </summary>
    public IFunction ActivationFunction
    {
      get { return m_ActivationFunction; }
      set { m_ActivationFunction = value; }
    }

    /// <summary>
    /// Indexer for networl layers
    /// </summary>
    public NeuralLayer this[int idx] { get { return SubNodes[idx]; } }

    /// <summary>
    /// Creates new neural layer and adds the result in the end of layer list
    /// </summary>
    public NeuralLayer CreateLayer()
    {
      var len = SubNodes.Length;
      var inputDim = (len <= 0) ? InputDim : SubNodes[len-1].SubNodes.Length;

      var layer = new NeuralLayer(this, inputDim);
      this.AddSubNode(layer);
      return layer;
    }

    /// <summary>
    /// Calculates result array produced by network
    /// </summary>
    /// <param name="input">Input data array</param>
    public override double[] Calculate(double[] input)
    {
      if (InputDim != input.Length)
        throw new MLException("Incorret input vector dimension");

      double[] data = input;

      if (UseBias)
      {
        data = new double[input.Length + 1];
        Buffer.BlockCopy(input, 0, data, 0, input.Length);
        data[input.Length] = 1.0D;
      }

      return base.Calculate(data);
    }

    public override void DoBuild()
    {
      if (InputDim <= 0)
        throw new MLException("Input dimension has not been set");

      base.DoBuild();

      var inputDim = InputDim;
      ActivationFunction = ActivationFunction ?? Registry.ActivationFunctions.Identity;

      for (int i=0; i<SubNodes.Length; i++)
      {
        var layer = SubNodes[i];
        if (layer.ActivationFunction==null)
          layer.ActivationFunction = ActivationFunction;
        for (int j=0; j<layer.SubNodes.Length; j++)
        {
          var node = layer.SubNodes[j];
          if (node.ActivationFunction==null)
            node.ActivationFunction = layer.ActivationFunction;
        }
      }
    }
  }
}

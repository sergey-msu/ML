using System;
using System.Collections.Generic;
using ML.Contracts;
using ML.Core.ComputingNetworks;
using ML.NeuralMethods.Contracts;

namespace ML.NeuralMethods.Networks
{
  /// <summary>
  /// Represents artificial neural network: set of layers with neuron nodes and weighted connections
  /// </summary>
  public class NeuralNetwork<TOut> : SequenceOutputNode<double[], TOut, NeuralLayer>, INeuralNetwork
  {
    public NeuralNetwork()
    {
    }

    /// <summary>
    /// If true, adds artificial +1 input value in the and of input data array
    /// </summary>
    public bool UseBias { get; set; }

    /// <summary>
    /// Layer activation function. If null, the network's activation function will be used
    /// </summary>
    public IFunction ActivationFunction { get; set; }

    /// <summary>
    /// Creates new neural layer and adds the result in the end of layer list
    /// </summary>
    public NeuralLayer CreateHiddenLayer()
    {
      var layer = new NeuralLayer(this);
      this.AddHiddenNode(layer);
      return layer;
    }

    /// <summary>
    /// Calculates result array produced by network
    /// </summary>
    /// <param name="input">Input data array</param>
    public override TOut Calculate(double[] input)
    {
      double[] data = input;

      if (UseBias)
      {
        data = new double[input.Length + 1];
        Buffer.BlockCopy(input, 0, data, 0, input.Length);
        data[input.Length] = 1.0D;
      }

      return base.Calculate(data);
    }

    object IComputingNode<double[], object>.Calculate(double[] input)
    {
      return Calculate(input);
    }
  }
}

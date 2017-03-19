using System;
using System.Collections.Generic;
using ML.Contracts;
using ML.Core;
using ML.Core.ComputingNetworks;

namespace ML.NeuralMethods.Model
{
  /// <summary>
  /// Represents feedforward artificial neural network: set of layers with neuron nodes and weighted connections
  /// </summary>
  public class NeuralNetwork : SequenceNode<double[], NeuralLayer>
  {
    #region Fields

    private int m_InputDim;
    private IActivationFunction m_ActivationFunction;

    #endregion

    #region .ctor

    public NeuralNetwork(int inputDim)
    {
      if (inputDim <= 0)
        throw new MLException("NeuralLayer.ctor(inputDim<=0)");

      m_InputDim = inputDim;
    }

    #endregion

    #region Properties

    /// <summary>
    /// Total count of network layers (hidden + output)
    /// </summary>
    public int LayerCount { get { return SubNodes.Length; } }

    /// <summary>
    /// Dimension of input vector
    /// </summary>
    public int InputDim { get { return m_InputDim; } }

    /// <summary>
    /// Layer activation function. If null, the network's activation function will be used
    /// </summary>
    public IActivationFunction ActivationFunction
    {
      get { return m_ActivationFunction; }
      set { m_ActivationFunction = value; }
    }

    /// <summary>
    /// Indexer for network layers
    /// </summary>
    public NeuralLayer this[int idx] { get { return SubNodes[idx]; } }

    #endregion

    #region Public

    /// <summary>
    /// Adds new neural layer in the end of layer list
    /// </summary>
    public virtual void AddLayer(NeuralLayer layer)
    {
      if (layer==null)
        throw new MLException("Layer can not be null");

      var prevOutputDim = (LayerCount == 0) ? InputDim : this[LayerCount-1].NeuronCount;
      if (layer.InputDim != prevOutputDim)
        throw new MLException("Layer input dimension differs with layer's one");

      this.AddSubNode(layer);
    }

    /// <summary>
    /// Randomizes network weights
    /// </summary>
    public virtual void RandomizeParameters(int seed=0)
    {
      foreach (var layer in SubNodes)
        layer.RandomizeParameters(seed);
    }

    /// <summary>
    /// Calculates result array produced by network
    /// </summary>
    /// <param name="input">Input data array</param>
    public override double[] Calculate(double[] input)
    {
      if (InputDim != input.Length)
        throw new MLException("Incorrect input vector dimension");

      return base.Calculate(input);
    }

    public override void DoBuild()
    {
      if (InputDim <= 0)
        throw new MLException("Input dimension has not been set");

      foreach (var layer in this.SubNodes)
        layer.ActivationFunction = layer.ActivationFunction ?? ActivationFunction;

      base.DoBuild();
    }

    #endregion
  }
}

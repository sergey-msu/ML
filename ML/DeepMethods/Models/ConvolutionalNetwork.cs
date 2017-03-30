using System;
using System.Collections.Generic;
using ML.Contracts;
using ML.Core.ComputingNetworks;
using ML.Core;

namespace ML.DeepMethods.Models
{
  /// <summary>
  /// Represents feedforward CNN: set of convolutional layers with shared weights along with the pooling layers
  /// </summary>
  public class ConvolutionalNetwork : SequenceNode<double[,,], DeepLayerBase>
  {
    #region Fields

    private IActivationFunction m_ActivationFunction;
    private int m_InputSize;
    private int m_InputDepth;

    #endregion

    public ConvolutionalNetwork(int inputDepth, int inputSize)
    {
      if (inputDepth <= 0)
        throw new MLException("ConvolutionalNetwork.ctor(inputDepth<=0)");
      if (inputSize <= 0)
        throw new MLException("ConvolutionalNetwork.ctor(inputSize<=0)");

      m_InputDepth = inputDepth;
      m_InputSize = inputSize;
    }

    #region Properties

    /// <summary>
    /// Total count of network layers (hidden + output)
    /// </summary>
    public int LayerCount { get { return SubNodes.Length; } }

    /// <summary>
    /// Size of square input matrix
    /// </summary>
    public int InputSize { get { return m_InputSize; } }

    /// <summary>
    /// Count of input channels
    /// </summary>
    public int InputDepth { get { return m_InputDepth; } }

    /// <summary>
    /// Activation function. If null, the layer's activation function will be used
    /// </summary>
    public IActivationFunction ActivationFunction
    {
      get { return m_ActivationFunction; }
      set { m_ActivationFunction = value; }
    }

    /// <summary>
    /// Indexer for network layers
    /// </summary>
    public DeepLayerBase this[int idx] { get { return SubNodes[idx]; } }

    #endregion

    /// <summary>
    /// Adds new deep layer in the end of layer list
    /// </summary>
    public virtual ConvolutionalNetwork AddLayer(DeepLayerBase layer)
    {
      if (layer==null)
        throw new MLException("Layer can not be null");

      var player = (LayerCount == 0) ? null : this[LayerCount-1];
      var pdepth = (player==null) ? InputDepth : player.OutputDepth;
      var psize  = (player==null) ? InputSize : player.OutputSize;
      if (layer.InputDepth != pdepth)
        throw new MLException("Layer input depth differs with prevous layer's one / input");
      if (layer.InputSize != psize)
        throw new MLException("Layer input size differs with prevous layer's one / input");

      this.AddSubNode(layer);

      return this;
    }

    /// <summary>
    /// Randomizes layer parameters (i.e. kernel weights, biases etc.)
    /// </summary>
    public virtual void RandomizeParameters(int seed)
    {
      foreach (var layer in this.SubNodes)
        layer.RandomizeParameters(seed);
    }

    public override void DoBuild()
    {
      foreach (var layer in this.SubNodes)
        if (layer.ActivationFunction == null) layer.ActivationFunction = ActivationFunction;

      base.DoBuild();
    }

    #region Serialization

    public void Serialize(System.IO.Stream stream)
    {
      var serializer = new NFX.Serialization.Slim.SlimSerializer(NFX.IO.SlimFormat.Instance);
      serializer.Serialize(stream, this);
    }

    public static ConvolutionalNetwork Deserialize(System.IO.Stream stream)
    {
      var serializer = new NFX.Serialization.Slim.SlimSerializer(NFX.IO.SlimFormat.Instance);
      return (ConvolutionalNetwork)serializer.Deserialize(stream);
    }

    #endregion
  }
}

using System;
using ML.Contracts;
using ML.Core.ComputingNetworks;
using ML.Core;

namespace ML.DeepMethods.Models
{
  /// <summary>
  /// Represents feedforward sequentional CNN: set of convolutional layers with shared weights along with the pooling layers
  /// </summary>
  public class ConvNet : SequenceNode<double[][,], DeepLayerBase>
  {
    #region Fields

    private InputLayer m_InputLayer;

    private IActivationFunction m_ActivationFunction;
    private int  m_InputHeight;
    private int  m_InputWidth;
    private int  m_InputDepth;
    private bool m_IsTraining;

    private double[][] m_Weights;

    #endregion

    public ConvNet(int inputDepth,
                   int inputSize,
                   IActivationFunction activation = null)
      : this(inputDepth, inputSize, inputSize, activation)
    {
    }

    public ConvNet(int inputDepth,
                   int inputHeight,
                   int inputWidth,
                   IActivationFunction activation = null)
    {
      if (inputDepth <= 0)
        throw new MLException("ConvolutionalNetwork.ctor(inputDepth<=0)");
      if (inputHeight <= 0)
        throw new MLException("ConvolutionalNetwork.ctor(inputHeight<=0)");
      if (inputWidth <= 0)
        throw new MLException("ConvolutionalNetwork.ctor(inputWidth<=0)");

      m_InputDepth  = inputDepth;
      m_InputHeight = inputHeight;
      m_InputWidth  = inputWidth;
      m_ActivationFunction = activation;
    }

    #region Properties

    /// <summary>
    /// True for using the network in training mode
    /// </summary>
    public bool IsTraining
    {
      get { return m_IsTraining; }
      set
      {
        m_IsTraining=value;
        foreach (var layer in SubNodes)
          layer.IsTraining = value;
      }
    }

    /// <summary>
    /// All trainable network weights
    /// </summary>
    public double[][] Weights { get { return m_Weights; } }

    /// <summary>
    /// Fictive input layer
    /// </summary>
    public InputLayer InputLayer { get { return m_InputLayer; } }

    /// <summary>
    /// Total count of network layers (hidden + output)
    /// </summary>
    public int LayerCount { get { return SubNodes.Length; } }

    /// <summary>
    /// Count of input channels
    /// </summary>
    public int InputDepth { get { return m_InputDepth; } }

    /// <summary>
    /// Height of input matrix
    /// </summary>
    public int InputHeight { get { return m_InputHeight; } }

    /// <summary>
    /// Width of input matrix
    /// </summary>
    public int InputWidth { get { return m_InputWidth; } }

    /// <summary>
    /// Activation function. If null, the layer's activation function will be used
    /// </summary>
    public IActivationFunction ActivationFunction
    {
      get { return m_ActivationFunction; }
      set { m_ActivationFunction = value; }
    }

    /// <summary>
    /// Indexer for network layers.
    /// "-1" returns fictive input layer
    /// </summary>
    public DeepLayerBase this[int idx]
    {
      get { return (idx==-1) ? m_InputLayer : SubNodes[idx]; }
    }

    #endregion

    /// <summary>
    /// Adds new deep layer in the end of layer list
    /// </summary>
    public virtual void AddLayer(DeepLayerBase layer)
    {
      if (layer==null)
        throw new MLException("Layer can not be null");

      this.AddSubNode(layer);
    }

    public void _Build()
    {
      // build layer sizes

      m_InputLayer = new InputLayer
      {
        InputDepth  = InputDepth,
        InputHeight = InputHeight,
        InputWidth  = InputWidth
      };
      m_InputLayer._Build();

      var depth  = InputDepth;
      var height = InputHeight;
      var width  = InputWidth;

      foreach (var layer in this.SubNodes)
      {
        if (layer.ActivationFunction == null) layer.ActivationFunction = ActivationFunction;
        layer.IsTraining  = IsTraining;
        layer.InputDepth  = depth;
        layer.InputHeight = height;
        layer.InputWidth  = width;
        layer._Build();

        depth  = layer.OutputDepth;
        height = layer.OutputHeight;
        width  = layer.OutputWidth;
      }

      // build network parameters

      var lcount = LayerCount;
      m_Weights = new double[lcount][];

      for (int i=0; i<lcount; i++)
      {
        var layer = this[i];
        m_Weights[i] = layer.Weights;
      }
    }

    /// <summary>
    /// Randomizes layer parameters (i.e. kernel weights, biases etc.)
    /// </summary>
    public virtual void RandomizeParameters(int seed)
    {
      foreach (var layer in this.SubNodes)
        layer.RandomizeParameters(seed);
    }

    public void Calculate(double[][,] input, double[][][,] result)
    {
      for (int i=0; i<SubNodes.Length; i++)
      {
        var prev = (i>0) ? result[i-1] : input;
        SubNodes[i].Calculate(prev, result[i]);
      }
    }

    public override double[][,] Calculate(double[][,] input)
    {
      m_InputLayer.Calculate(input);
      return base.Calculate(input);
    }

    #region Serialization

    public void Serialize(System.IO.Stream stream)
    {
      var serializer = new NFX.Serialization.Slim.SlimSerializer(NFX.IO.SlimFormat.Instance);
      serializer.Serialize(stream, this);
    }

    public static ConvNet Deserialize(System.IO.Stream stream)
    {
      var serializer = new NFX.Serialization.Slim.SlimSerializer(NFX.IO.SlimFormat.Instance);
      return (ConvNet)serializer.Deserialize(stream);
    }

    #endregion
  }
}

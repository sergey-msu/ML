using System;
using System.Collections.Generic;
using System.Linq;
using ML.Core;
using ML.NeuralMethods.Models;
using ML.Contracts;

namespace ML.NeuralMethods.Algorithms
{
  /// <summary>
  /// Multi-layer neural network training algorithm.
  /// Using Backpropagation principle as a core.
  /// </summary>
  public class BackpropAlgorithm : NeuralNetAlgorithmBase
  {
    #region Inner

    public enum StopCriteria
    {
      FullLoop  = 0,
      ErrorFunc = 1,
      StepMin   = 2
    }

    #endregion

    #region CONST

    public const int    DFT_EPOCH_COUNT = 1;
    public const int    DFT_BATCH_SIZE = 1;
    public const double DFT_LEARNING_RATE = 0.1D;
    public const StopCriteria DTF_STOP_CRITERIA = StopCriteria.FullLoop;

    #endregion

    #region Fields

    private int m_EpochLength;

    private ILossFunction m_LossFunction;
    private StopCriteria m_Stop;

    private int    m_BatchSize;
    private int    m_InputDim;
    private int    m_OutputDim;
    private int    m_EpochCount;
    private double m_LearningRate;

    private double m_IterErrorValue;
    private double m_PrevErrorValue;
    private double m_ErrorValue;
    private double m_ErrorDelta;
    private double m_ErrorStopDelta;

    private double m_Step2;
    private double m_StepStopValue;

    private int m_Epoch;
    private int m_Iteration;
    private int m_Batch;

    private double[][]  m_Errors;
    private double[][,] m_Gradient;

    #endregion

    #region .ctor

    public BackpropAlgorithm(NeuralNetwork net) : base(net)
    {
      init();
    }

    #endregion

    #region Events

    public event EventHandler EpochEndedEvent;

    #endregion

    #region Properties

    public override string ID { get { return "MLP_BP"; } }
    public override string Name { get { return "MLP Neural Network with Backpropagation"; } }

    public int InputDim          { get { return m_InputDim; } }
    public int OutputDim         { get { return m_OutputDim; } }
    public double IterErrorValue { get { return m_IterErrorValue; } }
    public double ErrorValue     { get { return m_ErrorValue; } }
    public double ErrorDelta     { get { return m_ErrorDelta; } }
    public double Step2          { get { return m_Step2; } }

    public ILossFunction LossFunction
    {
      get { return m_LossFunction; }
      set { m_LossFunction = value; }
    }

    public int EpochCount
    {
      get { return m_EpochCount; }
      set
      {
        if (value <= 0)
          throw new MLException("Epoch count must be positive");
        m_EpochCount = value;
      }
    }

    public int BatchSize
    {
      get { return m_BatchSize; }
      set
      {
        if (value <= 0)
          throw new MLException("Batch size must be positive");
        m_BatchSize = value;
      }
    }

    public double LearningRate
    {
      get { return m_LearningRate; }
      set
      {
        if (value <= 0)
          throw new MLException("Learning rate must be positive");
        m_LearningRate = value;
      }
    }

    public StopCriteria Stop
    {
      get { return m_Stop; }
      set { m_Stop = value; }
    }

    public double ErrorStopDelta
    {
      get { return m_ErrorStopDelta; }
      set
      {
        if (ErrorStopDelta < 0)
          throw new MLException("Error Stop Delta must be non-negative");
        m_ErrorStopDelta = value;
      }
    }

    public double StepStopValue
    {
      get { return m_StepStopValue; }
      set
      {
        if (StepStopValue < 0)
          throw new MLException("Step Stop Value must be non-negative");
        m_StepStopValue = value;
      }
    }

    public double[][]  Errors   { get { return m_Errors;  } }
    public double[][,] Gradient { get { return m_Gradient; } }

    public int Epoch { get { return m_Epoch; } }
    public int Iteration { get { return m_Iteration; } }
    public int Batch { get { return m_Batch; } }

    #endregion

    #region Public

    public void RunEpoch()
    {
      runEpoch(Net);
    }

    public void RunBatch(int skip, int take)
    {
      if (skip<0) throw new MLException("Skip value must be non-negative");
      if (take<=0) throw new MLException("Take value must be positive");

      runBatch(Net, TrainingSample.Subset(skip, take));
    }

    public void RunIteration(double[] data, double[] expected)
    {
      runIteration(Net, data, expected);
    }

    public void FlushGradient()
    {
      updateWeights(Net);
    }

    #endregion

    protected override void TrainImpl()
    {
      for (int epoch=0; epoch<m_EpochCount; epoch++)
      {
        runEpoch(Net);
        if (checkStopCriteria()) break;
      }
    }

    #region .pvt

    private void init()
    {
      m_EpochCount   = DFT_EPOCH_COUNT;
      m_BatchSize    = DFT_BATCH_SIZE;
      m_LearningRate = DFT_LEARNING_RATE;
      m_Stop         = DTF_STOP_CRITERIA;
      m_EpochLength  = TrainingSample.Count;
      m_InputDim     = Net.InputDim;
      m_OutputDim    = Net[Net.LayerCount-1].NeuronCount;

      m_Errors = new double[Net.LayerCount][];
      for (int i=0; i<Net.LayerCount; i++)
      {
        var ncount = Net[i].NeuronCount;
        m_Errors[i] = new double[ncount];
      }

      m_Gradient = new double[Net.LayerCount][,];
      for (int i=0; i<Net.LayerCount; i++)
      {
        var pcount = (i>0) ? Net[i-1].NeuronCount : m_InputDim;
        var lcount = Net[i].NeuronCount;
        m_Gradient[i] = new double[lcount, pcount+1]; // take bias into account
      }
    }

    private void runEpoch(NeuralNetwork net)
    {
      // loop on batches
      foreach (var batch in TrainingSample.Batch(m_BatchSize))
      {
        runBatch(net, batch);
      }

      // update epoch stats
      m_Epoch++;
      m_Iteration = 0;
      m_Batch = 0;

      if (EpochEndedEvent != null) EpochEndedEvent(this, EventArgs.Empty);
    }

    private void runBatch(NeuralNetwork net, MultiRegressionSample<double[]> sampleBatch)
    {
      // loop on batch
      foreach (var pdata in sampleBatch)
      {
        runIteration(net, pdata.Key, pdata.Value);
      }

      updateWeights(net);

      // update batch stats
      m_PrevErrorValue = m_ErrorValue;
      m_ErrorValue = m_IterErrorValue/sampleBatch.Count;
      m_ErrorDelta = m_ErrorValue-m_PrevErrorValue;
      m_IterErrorValue = 0.0D;

      m_Batch++;
    }

    private void runIteration(NeuralNetwork net, double[] input, double[] expected)
    {
      // forward calculation
      var serr = feedForward(net, input, expected);

      // error backpropagation
      var lcount = net.LayerCount;
      for (int i=lcount-1; i>=0; i--)
      {
        feedBackward(net, net[i], i, input);
      }

      // update iter stats
      m_IterErrorValue += serr;
      m_Iteration++;
    }

    private double feedForward(NeuralNetwork net, double[] input, double[] expected)
    {
      var output = net.Calculate(input);
      var llayer = net[net.LayerCount-1];
      var errors = m_Errors[net.LayerCount-1];

      for (int j=0; j<m_OutputDim; j++)
      {
        var neuron = llayer[j];
        var ej = m_LossFunction.Derivative(j, output, expected);
        errors[j] = ej * neuron.Derivative;
      }

      return m_LossFunction.Value(output, expected);
    }

    private void feedBackward(NeuralNetwork net, NeuralLayer layer, int lidx, double[] input)
    {
      var errors  = m_Errors[lidx];
      var lgrad   = m_Gradient[lidx];
      var ncount  = layer.NeuronCount;
      var player  = (lidx>0) ? net[lidx-1] : null;
      var pcount  = (lidx>0) ? player.NeuronCount : m_InputDim;
      var perrors = (lidx>0) ? m_Errors[lidx-1] : null;

      // save "errors" in previous layer for future use
      if (lidx > 0)
        for (int h=0; h<pcount; h++)
        {
          var pneuron = player[h];
          if (!pneuron.LastRetained)
          {
            perrors[h] = 0;
            continue;
          }

          var eh = 0.0D;
          for (int j=0; j<ncount; j++)
          {
            var neuron = layer[j];
            if (!neuron.LastRetained) continue;

            eh += errors[j] * neuron[h];
          }

          perrors[h] = eh * pneuron.Derivative;
        }

      // accumulate gradient
      for (int j=0; j<ncount; j++)
      {
        var neuron = layer[j];
        if (!neuron.LastRetained) continue;

        var gj = errors[j];

        for (int h=0; h<pcount; h++)
        {
          if ((lidx != 0) && !player[h].LastRetained) continue;

          var value = (lidx>0) ? player[h].Value : input[h];
          var dwj = gj * value;
          lgrad[j, h] += dwj;
        }

        lgrad[j, pcount] += gj;
      }
    }

    private void updateWeights(NeuralNetwork net)
    {
      var lcount = net.LayerCount;
      var step2 = 0.0D;

      for (int i=lcount-1; i>=0; i--)
      {
        var lgrad  = m_Gradient[i];
        var layer  = net[i];
        var ncount = layer.NeuronCount;
        var player = (i>0) ? net[i-1] : null;
        var pcount = (i>0) ? player.NeuronCount : m_InputDim;
        double dw;

        for (int j=0; j<ncount; j++)
        {
          var neuron = layer[j];
          for (int h=0; h<pcount; h++)
          {
            dw = -m_LearningRate*lgrad[j, h];
            step2 += dw*dw;
            neuron[h] += dw;
          }

          dw = -m_LearningRate*lgrad[j, pcount];
          step2 += dw*dw;
          neuron.Bias += dw;
        }

        Array.Clear(lgrad, 0, lgrad.Length);
      }

      m_Step2 = step2;
    }

    private bool checkStopCriteria()
    {
      switch (Stop)
      {
        case StopCriteria.FullLoop:  return false;
        case StopCriteria.ErrorFunc: return Math.Abs(m_ErrorDelta) < m_ErrorStopDelta;
        case StopCriteria.StepMin:   return m_Step2 < StepStopValue;
        default: throw new MLException("Unknown stop criteria");
      }
    }

    #endregion
  }
}

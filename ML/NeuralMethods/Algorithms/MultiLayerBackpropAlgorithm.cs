using System;
using System.Collections.Generic;
using System.Linq;
using ML.Core;
using ML.NeuralMethods.Model;
using ML.Contracts;

namespace ML.NeuralMethods.Algorithms
{
  public class MultiLayerBackpropAlgorithm : NeuralNetworkAlgorithmBase
  {
    #region Inner

    public enum StopCriteria
    {
      FullLoop = 0,
      ErrorMin = 1,
      StepMin = 2
    }

    #endregion

    #region CONST

    public const bool   DFT_USE_BIAS = true;
    public const int    DFT_EPOCH_COUNT = 1;
    public const double DFT_LEARNING_RATE = 0.1D;
    public const double DFT_ERROR_DELTA = 0.0D;
    public const double DFT_STOP_STEP_LEVEL = 0.0D;
    public const StopCriteria DTF_STOP_CRITERIA = StopCriteria.FullLoop;
    public static readonly IFunction DFT_ACTIVATION_FUNCTION = Registry.ActivationFunctions.Identity;

    #endregion

    #region Fields

    private static double[][] m_BackErrors;

    private NeuralNetwork m_Network;
    private int[] m_Structure;
    private Dictionary<Class, double[]> m_ExpectedOutputs;

    private IFunction m_ActivationFunction;
    private int    m_InputDim;
    private int    m_OutputDim;
    private double m_PrevError;
    private double m_Error;
    private double m_Step;
    private bool   m_UseBias;
    private int    m_EpochCount;
    private double m_LearningRate;
    private StopCriteria m_Stop;
    private double m_ErrorDelta;
    private double m_StopStepLevel;

    #endregion

    public MultiLayerBackpropAlgorithm(ClassifiedSample classifiedSample, NeuralNetwork net)
      : base(classifiedSample)
    {
      if (net==null)
        throw new MLException("Network can not be null");

      m_Network = net;
      initParams();
    }

    public MultiLayerBackpropAlgorithm(ClassifiedSample classifiedSample, int[] structure)
      : base(classifiedSample)
    {
      if (structure==null)
        throw new MLException("Network structure can not be null");

      m_Structure = structure;
      initParams();
    }


    #region Properties

    public override string ID { get { return "BP_NN"; } }
    public override string Name { get { return "Backpropagation Neural Network"; } }

    public int InputDim { get { return m_InputDim; } }
    public int OutputDim { get { return m_OutputDim; } }
    public double Error { get { return m_Error; } }
    public double Step { get { return m_Step; } }

    public StopCriteria Stop
    {
      get { return m_Stop; }
      set { m_Stop = value; }
    }
    public double ErrorDelta
    {
      get { return m_ErrorDelta; }
      set { m_ErrorDelta = value; }
    }
    public double StopStepLevel
    {
      get { return m_StopStepLevel; }
      set { m_StopStepLevel = value; }
    }
    public bool UseBias
    {
      get { return m_UseBias; }
      set { m_UseBias = value; }
    }
    public IFunction ActivationFunction
    {
      get { return m_ActivationFunction; }
      set { m_ActivationFunction = value; }
    }
    public int EpochCount
    {
      get { return m_EpochCount; }
      set { m_EpochCount = value; }
    }
    public double LearningRate
    {
      get { return m_LearningRate; }
      set { m_LearningRate = value; }
    }

    #endregion

    protected override NeuralNetwork DoTrain()
    {
      prepareData();
      checkParams();
      var net = m_Network ?? constructNetwork();
      checkNetwork();
      doTrain(net);

      return net;
    }

    #region .pvt

    private void initParams()
    {
      ActivationFunction = DFT_ACTIVATION_FUNCTION;
      UseBias            = DFT_USE_BIAS;
      EpochCount         = DFT_EPOCH_COUNT;
      LearningRate       = DFT_LEARNING_RATE;
      Stop               = DTF_STOP_CRITERIA;
      StopStepLevel      = DFT_STOP_STEP_LEVEL;
      LearningRate       = DFT_LEARNING_RATE;
      ErrorDelta         = DFT_ERROR_DELTA;
    }

    private void checkParams()
    {
      if (m_Network==null && m_Structure==null)
        throw new MLException("Either network or network structure must be set");
      if (m_Network!=null && m_Structure!=null)
        throw new MLException("Network and network structucture can not be set both at the same time");
      if (m_Structure!=null && m_Structure.Length<2)
        throw new MLException("At least two lengths must be present in network structure description: [0] - input dimension, [last] - output dimension");

      if (ActivationFunction==null) throw new MLException("Activaltion function is null");
      if (InputDim <= 0)     throw new MLException("Input dimension must be positive");
      if (OutputDim <= 0)    throw new MLException("Output dimension nust be positive");
      if (EpochCount <= 0)   throw new MLException("Epoch count must be positive");
      if (LearningRate <= 0) throw new MLException("Learning rate must be positive");
    }

    private void prepareData()
    {
      // parameters

      if (m_Network==null)
      {
        var lcount = m_Structure.Length-1;
        m_InputDim = m_Structure[0];
        m_OutputDim = m_Structure[lcount];
        m_BackErrors = new double[lcount][];
        for (int i=0; i<lcount; i++)
          m_BackErrors[i] = new double[m_Structure[i+1]];
      }
      else
      {
        var lcount = m_Network.LayerCount;
        m_InputDim = m_Network.InputDim;
        m_OutputDim = m_Network[lcount-1].NeuronCount;
        m_BackErrors = new double[lcount][];
        for (int i=0; i<lcount; i++)
          m_BackErrors[i] = new double[m_Network[i].NeuronCount];
      }

      // expected outputs

      m_ExpectedOutputs = new Dictionary<Class, double[]>();
      var count = Classes.Count;
      if (count != OutputDim)
        throw new MLException("Number of classes must be equal to dimetsion of output vector");

      for (int i=0; i<count; i++)
      {
        var cls = Classes.FirstOrDefault(p => (int)p.Value.Value == i+1).Value;
        if (cls==null)
          throw new MLException(string.Format("There is no class with value {0}. It is neccessary to have full set of classes with values from 0 to {1}", i, count));

        var output = new double[count];
        output[i] = 1.0D;
        m_ExpectedOutputs[cls] = output;
      }
    }

    private void checkNetwork()
    {
      // TODO : if NN was passed from outside we need to check its structure for consistency
    }

    private NeuralNetwork constructNetwork()
    {
      var net = new NeuralNetwork();
      net.InputDim = m_InputDim;
      net.UseBias = UseBias;
      net.ActivationFunction = ActivationFunction;

      var lcount = m_Structure.Length-1;
      for (int i=1; i<=lcount; i++)
      {
        var dim = m_Structure[i];
        var layer = net.CreateLayer();
        for (int j=0; j<dim; j++)
          layer.CreateNeuron<FullNeuron>();
      }

      net.Build();

      return net;
    }

    private void doTrain(NeuralNetwork net)
    {
      for (int epoch=0; epoch<m_EpochCount; epoch++)
      {
        runEpoch(net);
        if (checkStopCriteria()) break;
      }
    }

    private void runEpoch(NeuralNetwork net)
    {
      var iters = TrainingSample.Count;
      double ierr2;
      double terr2 = 0.0D;
      double istep2;
      double tstep2 = 0.0D;

      foreach (var pdata in TrainingSample)
      {
        runIter(net, pdata.Key, pdata.Value, out ierr2, out istep2);
        terr2 += ierr2;
        tstep2 = Math.Max(tstep2, istep2);
      }

      m_PrevError = m_Error;
      m_Error = terr2/iters;
      m_Step = tstep2;
    }

    private void runIter(NeuralNetwork net,
                         Point data, Class cls,
                         out double ierr2, out double istep2)
    {
      var lcount = net.LayerCount;
      var serr2 = 0.0D;
      var sstep2 = 0.0D;

      // forward calculation
      var result = net.Calculate(data);
      var expect = m_ExpectedOutputs[cls];
      var errors = m_BackErrors[lcount-1];
      for (int j=0; j<m_OutputDim; j++)
      {
        var ej = result[j] - expect[j];
        errors[j] = ej;
        serr2 += ej*ej;
      }

      // error backpropagation
      for (int i=lcount-1; i>=0; i--)
      {
        // calculate current layer errors
        var layer = net[i];
        var ncount = layer.NeuronCount;
        var derrors = new double[ncount];
        errors = m_BackErrors[i];

        for (int j=0; j<ncount; j++)
        {
          var neuron = layer[j];
          var ej = errors[j];
          var oj = neuron.Value;
          derrors[j] = ej * neuron.ActivationFunction.Derivative(oj);
        }

        // save errors for future use
        int pcount;
        NeuralLayer prev;
        if (i>0)
        {
          prev = net[i-1];
          pcount = prev.NeuronCount;
          var perrors = m_BackErrors[i-1];
          for (int h=0; h<pcount; h++)
          {
            var neuron = prev[h];
            var perror = 0.0D;
            for (int j=0; j<ncount; j++)
              perror += derrors[j] * neuron[h];
            perrors[h] = perror;
          }
        }
        else
        {
          pcount = m_InputDim;
          prev = null;
        }

        // gradient step - weights update
        for (int j=0; j<ncount; j++)
        {
          var neuron = layer[j];
          var dj = m_LearningRate * derrors[j];

          for (int h=0; h<pcount; h++)
          {
            var value = (i==0) ? data[h] : prev[h].Value;
            var dwj = dj * value;
            neuron[h] -= dwj;
            sstep2 += dwj*dwj;
          }
          if (UseBias)
          {
            neuron[pcount] -= dj;
            sstep2 += dj*dj;
          }
        }
      }

      ierr2 = serr2/2;
      istep2 = sstep2;
    }

    private bool checkStopCriteria()
    {
      switch (Stop)
      {
        case StopCriteria.FullLoop: return false;
        case StopCriteria.ErrorMin: return Math.Abs(m_Error-m_PrevError) < ErrorDelta;
        case StopCriteria.StepMin:  return m_Step < StopStepLevel;
        default: throw new MLException("Unknown stop citeria");
      }
    }

    #endregion
  }
}

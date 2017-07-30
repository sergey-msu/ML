﻿using System;
using ML.Contracts;

namespace ML.Core.ActivationFunctions
{
  /// <summary>
  /// Binary Step Activation Function
  /// </summary>
  public sealed class StepActivation : IFunction
  {
    public string Name { get { return "STEP"; } }


    public double Value(double r)
    {
      return r < 0 ? 0 : 1;
    }

    public double Derivative(double r)
    {
      return 0;
    }

    public double DerivativeFromValue(double y)
    {
      return 0;
    }
  }

  /// <summary>
  /// Binary Step Activation Function
  /// </summary>
  public sealed class ShiftedStepActivation : IActivationFunction
  {
    private readonly double m_Threshold;

    public ShiftedStepActivation(double threshold)
    {
      m_Threshold = threshold;
    }

    public string Name { get { return "SSTEP"; } }

    public double Threshold { get { return m_Threshold; } }

    public double Value(double r)
    {
      return r < m_Threshold ? 0 : 1;
    }

    public double Derivative(double r)
    {
      return 0;
    }

    public double DerivativeFromValue(double y)
    {
      return 0;
    }
  }
}

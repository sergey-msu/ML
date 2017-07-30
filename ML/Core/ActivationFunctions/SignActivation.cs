﻿using System;
using ML.Contracts;

namespace ML.Core.ActivationFunctions
{
  /// <summary>
  /// Signum Activation Function
  /// </summary>
  public class SignActivation : IActivationFunction
  {
    public SignActivation(double zeroValue = 1)
    {
      ZeroValue = zeroValue;
    }

    public readonly double ZeroValue;
    public string Name { get { return "SIGN"; } }


    public double Value(double r)
    {
      if (r==0) return ZeroValue;
      return (r<0) ? -1 : 1;
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

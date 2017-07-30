﻿using System;
using ML.Contracts;

namespace ML.Core.ActivationFunctions
{
  /// <summary>
  /// Rectified linear unit Activation Function
  /// </summary>
  public class ReLUActivation : IActivationFunction
  {
    public string Name { get { return "RELU"; } }


    public double Value(double r)
    {
      return (r < 0) ? 0 : r;
    }

    public double Derivative(double r)
    {
      return (r < 0) ? 0 : 1;
    }

    public double DerivativeFromValue(double y)
    {
      if (y<0)
        throw new MLException("ReLU value must be non negative");

      return (y==0) ? 0 : 1;
    }
  }
}

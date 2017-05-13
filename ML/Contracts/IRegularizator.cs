using System;

namespace ML.Contracts
{
  public interface IRegularizator
  {
    double Value(double[][] weights);

    void Apply(double[][] gradients, double[][] weights);
  }
}

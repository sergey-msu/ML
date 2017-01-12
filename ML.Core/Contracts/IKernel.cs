namespace ML.Core.Contracts
{
  public interface IKernel
  {
    string Name { get; }

    float Calculate(float r);
  }
}

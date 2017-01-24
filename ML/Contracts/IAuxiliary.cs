namespace ML.Contracts
{
  /// <summary>
  /// COntract for named entities
  /// </summary>
  public interface INamed
  {
    /// <summary>
    /// Name
    /// </summary>
    string Name { get; }
  }

  /// <summary>
  /// Contract for named and mnemonicaly named (supplied with some ID) entity
  /// </summary>
  public interface IMnemonicNamed : INamed
  {
    /// <summary>
    /// Mnemonic ID
    /// </summary>
    string ID { get; }
  }
}

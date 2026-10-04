# Contributing

Thanks for contributing to Halloween AI.

## Great contribution areas
- person ReID and cross-camera matching
- animal/costume detection
- multi-person handling
- dashboard/UI improvements
- camera compatibility
- dispenser and sensor integrations
- MQTT/Home Assistant support
- documentation and testing

## Workflow
1. Fork the repository.
2. Create a focused branch.
3. Develop in simulation mode where practical.
4. Test failure/offline cases as well as the happy path.
5. Open a pull request explaining the change and testing performed.

Please include OS, Python version, relevant camera/hardware details and whether tests used simulation or real hardware.

Do not commit credentials, private visitor imagery, personal data or private deployment configuration.

The project's reliability philosophy favours serving an uncertain genuine visitor over demanding perfect AI recognition. Detection changes should keep that behaviour explicit and configurable.

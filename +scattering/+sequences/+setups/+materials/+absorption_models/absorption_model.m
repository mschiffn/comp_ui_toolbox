%
% superclass for all absorption models
%
% author: Martin F. Schiffner
% date: 2016-08-25
% modified: 2021-05-10
%
classdef (Abstract) absorption_model

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        str_name                    % name of absorption model

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods (concrete)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = absorption_model( strs_name )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure cell array for strs_name
            if ~iscell( strs_name )
                strs_name = { strs_name };
            end

            %--------------------------------------------------------------
            % 2.) create absorption models
            %--------------------------------------------------------------
            % repeat default absorption model
            objects = repmat( objects, size( strs_name ) );

            % iterate absorption models
            for index_object = 1:numel( objects )

                % internal properties
                objects( index_object ).str_name = strs_name{ index_object };

            end % for index_object = 1:numel( objects )

        end % function objects = absorption_model( strs_name )

        %------------------------------------------------------------------
        % compute complex-valued wavenumbers
        %------------------------------------------------------------------
        function axes_k_tilde = compute_wavenumbers( absorption_models, axes_f )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class scattering.sequences.setups.materials.absorption_models.absorption_model
            if ~isa( absorption_models, 'scattering.sequences.setups.materials.absorption_models.absorption_model' )
                errorStruct.message = 'absorption_models must be scattering.sequences.setups.materials.absorption_models.absorption_model!';
                errorStruct.identifier = 'compute_wavenumbers:NoAbsorptionModels';
                error( errorStruct );
            end

            % ensure class math.sequence_increasing
% TODO: ensure physical_values.frequency
            if ~isa( axes_f, 'math.sequence_increasing' )
                axes_f = math.sequence_increasing( axes_f );
            end

            % ensure equal number of dimensions and sizes
            [ absorption_models, axes_f ] = auxiliary.ensureEqualSize( absorption_models, axes_f );

            %--------------------------------------------------------------
            % 2.) compute complex-valued wavenumbers
            %--------------------------------------------------------------
            % specify cell array
            axes_k_tilde = cell( size( absorption_models ) );

            % iterate time causal absorption models
            for index_object = 1:numel( absorption_models )

                % compute complex-valued wavenumbers (scalar)
                axes_k_tilde{ index_object } = compute_wavenumbers_scalar( absorption_models( index_object ), axes_f( index_object ) );

            end % for index_object = 1:numel( absorption_models )

            % create increasing sequences
            axes_k_tilde = math.sequence_increasing( axes_k_tilde );

        end % function axes_k_tilde = compute_wavenumbers( absorption_models, axes_f )

        %------------------------------------------------------------------
        % compute material transfer functions (MTFs)
        %------------------------------------------------------------------
        function MTFs = MTF( absorption_models, axes_f, distances )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure three arguments
            narginchk( 3, 3 );

            % ensure class scattering.sequences.setups.materials.absorption_models.absorption_model
            if ~isa( absorption_models, 'scattering.sequences.setups.materials.absorption_models.absorption_model' )
                errorStruct.message = 'absorption_models must be scattering.sequences.setups.materials.absorption_models.absorption_model!';
                errorStruct.identifier = 'MTF:NoAbsorptionModels';
                error( errorStruct );
            end

            % ensure class math.sequence_increasing
            if ~isa( axes_f, 'math.sequence_increasing' )
                errorStruct.message = 'axes_f must be math.sequence_increasing!';
                errorStruct.identifier = 'MTF:NoIncreasingSequences';
                error( errorStruct );
            end

            % ensure cell array for distances
            if ~iscell( distances )
                distances = { distances };
            end

            % ensure equal number of dimensions and sizes
            [ absorption_models, axes_f, distances ] = auxiliary.ensureEqualSize( absorption_models, axes_f, distances );

            %--------------------------------------------------------------
            % 2.) compute material transfer functions (MTFs)
            %--------------------------------------------------------------
            % compute wavenumbers
            axes_k_tilde = compute_wavenumbers( absorption_models, axes_f );

            % specify cell array for MTFs
            MTFs = cell( size( absorption_models ) );

            % iterate absorption models
            for index_object = 1:numel( absorption_models )
 
                %----------------------------------------------------------
                % a) check arguments
                %----------------------------------------------------------
                % ensure class physical_values.length
                if ~isa( distances{ index_object }, 'physical_values.length' )
                    errorStruct.message = 'distances must be physical_values.length!';
                    errorStruct.identifier = 'MTF:NoLengths';
                    error( errorStruct );
                end

                %----------------------------------------------------------
                % b) compute MTFs
                %----------------------------------------------------------
                MTFs{ index_object } = exp( -1j * axes_k_tilde( index_object ).members .* reshape( distances{ index_object }, [ 1, numel( distances{ index_object } ) ] ) );

            end % for index_object = 1:numel( absorption_models )

            % create signal matrices
            MTFs = processing.signal_matrix( axes_f, MTFs );

        end % function MTFs = MTF( absorption_models, axes_f, distances )

        %------------------------------------------------------------------
        % compute material impulse response functions (MIRFs)
        %------------------------------------------------------------------
        function MIRFs = MIRF( absorption_models, distances, varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure three arguments
            narginchk( 2, Inf );

            %--------------------------------------------------------------
            % 2.) compute material impulse response functions (MIRFs)
            %--------------------------------------------------------------
            % compute time durations
            absorption_models.c_ref
            % compute MTFs
            MTFs = MTF( absorption_models, axes_f, distances );

            % compute time-domain signal
            [ signal_matrices, N_samples_t ] = signal( MTFs, lbs_q, delta )
            signals = signal( MTFs );

        end % function MIRFs = MIRF( absorption_models, distances, varargin )

    end % methods

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (Abstract, protected, hidden)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Abstract, Access = protected, Hidden)

        %------------------------------------------------------------------
        % compute complex-valued wavenumbers (scalar)
        %------------------------------------------------------------------
        samples_k_tilde = compute_wavenumbers_scalar( absorption_model, axis_f )

	end % methods (Abstract, Access = protected, Hidden)

end % classdef (Abstract) absorption_model

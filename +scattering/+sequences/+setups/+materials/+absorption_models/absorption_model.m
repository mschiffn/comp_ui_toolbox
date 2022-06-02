%
% superclass for all absorption models
%
% author: Martin F. Schiffner
% date: 2016-08-25
% modified: 2021-05-15
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
        % TODO: compute group velocity
        %------------------------------------------------------------------

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
                    errorStruct.message = sprintf( 'distances{ %d } must be physical_values.length!', index_object );
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
        function MIRFs = MIRF( absorption_models, distances, intervals_f, T )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure at least three and at most four arguments
            narginchk( 3, 4 );

            % ensure cell array for distances
            if ~iscell( distances )
                distances = { distances };
            end

            % ensure equal subclasses of physical_values.length
            auxiliary.mustBeEqualSubclasses( 'physical_values.length', distances{ : } );

            % ensure class math.interval
            if ~isa( intervals_f, 'math.interval' )
                errorStruct.message = 'intervals_f must be math.interval!';
                errorStruct.identifier = 'MIRF:NoIntervals';
                error( errorStruct );
            end

            % ensure equal subclasses of physical_values.frequency
            auxiliary.mustBeEqualSubclasses( 'physical_values.frequency', intervals_f.lb );

            % ensure existence of nonempty T
            if nargin < 4 || isempty( T )
            end

            % ensure equal number of dimensions and sizes
            [ absorption_models, distances, intervals_f ] = auxiliary.ensureEqualSize( absorption_models, distances, intervals_f );

            %--------------------------------------------------------------
            % 2.) compute material impulse response functions (MIRFs)
            %--------------------------------------------------------------
            % extract frequency bounds
            f_lbs = reshape( [ intervals_f.lb ], size( absorption_models ) );
            f_ubs = reshape( [ intervals_f.ub ], size( absorption_models ) );

            % prevent zero frequencies
% TODO: define constant
            indicator = f_lbs < physical_values.hertz( 1 );
            f_lbs( indicator ) = physical_values.hertz( 1 );

            % compute phase velocities at frequency bounds
            axes_f = reshape( math.sequence_increasing( mat2cell( [ f_lbs( : ).'; f_ubs( : ).' ], 2, ones( 1, numel( absorption_models ) ) ) ), size( absorption_models ) );
            axes_k_tilde = compute_wavenumbers( absorption_models, axes_f );
            c_lb_ub = 2 * pi * cat( 2, axes_f.members ) ./ real( cat( 2, axes_k_tilde.members ) );
            c_min = min( c_lb_ub, [], 1 );

            % compute time durations
% TODO: permit argument to determine uniform duration
            T = physical_values.second( zeros( size( absorption_models ) ) );
            for index_object = 1:numel( absorption_models )
                T( index_object ) = max( distances{ index_object } / c_min( index_object ) );
            end

            % compute frequency axes
            q_lbs = ceil( f_lbs .* T );
            q_ubs = floor( f_ubs .* T );
            axes_f = math.sequence_increasing_regular_quantized( q_lbs, q_ubs, 1 ./ T );

            % compute MTFs
            MTFs = MTF( absorption_models, axes_f, distances );

            % compute time-domain signals
            MIRFs = signal( MTFs ); % , 0, 1 ./ ( 2 * intervals_f.ub )

        end % function MIRFs = MIRF( absorption_models, distances, intervals_f )

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

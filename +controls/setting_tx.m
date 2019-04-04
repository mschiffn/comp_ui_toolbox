%
% superclass for all transducer control settings in synthesis mode
%
% author: Martin F. Schiffner
% date: 2019-02-25
% modified: 2019-04-02
%
classdef setting_tx < controls.setting

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        excitation_voltages ( 1, : ) discretizations.signal_matrix	% voltages exciting the active channels

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = setting_tx( indices_active, impulse_responses, excitation_voltages )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            if nargin == 0
                indices_active = 1;
                impulse_responses = discretizations.signal_matrix( math.sequence_increasing_regular( 0, 0, physical_values.second ), 1 );
                excitation_voltages = discretizations.signal_matrix( math.sequence_increasing_regular( 0, 0, physical_values.second ), physical_values.voltage );
            end

            % ensure cell array for indices_active
            if ~iscell( indices_active )
                indices_active = { indices_active };
            end

            % ensure cell array for impulse_responses
            if ~iscell( impulse_responses )
                impulse_responses = { impulse_responses };
            end

            % ensure cell array for excitation_voltages
            if ~iscell( excitation_voltages )
                excitation_voltages = { excitation_voltages };
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( indices_active, excitation_voltages );

            %--------------------------------------------------------------
            % 2.) constructor of superclass
            %--------------------------------------------------------------
            objects@controls.setting( indices_active, impulse_responses );

            %--------------------------------------------------------------
            % 3.) set independent properties
            %--------------------------------------------------------------
            % iterate transducer control settings in synthesis mode
            for index_object = 1:numel( objects )

                switch class( excitation_voltages{ index_object } )

                    case 'discretizations.signal'

                        % multiple indices_active{ index_object } / single excitation_voltages{ index_object }
                        if ~isscalar( indices_active{ index_object } ) && isscalar( excitation_voltages{ index_object } )

                            samples = repmat( excitation_voltages{ index_object }.samples, [ numel( indices_active{ index_object } ), 1 ] );
                            excitation_voltages{ index_object } = discretizations.signal_matrix( excitation_voltages{ index_object }.axis, samples );

                        else

                            % ensure equal number of dimensions and sizes of cell array contents
                            auxiliary.mustBeEqualSize( indices_active{ index_object }, excitation_voltages{ index_object } );

                            % assemble signal_matrix for equal axes
                            if isequal( excitation_voltages{ index_object }.axis )
                                samples = reshape( excitation_voltages{ index_object }.samples, [ numel( excitation_voltages{ index_object } ), abs( excitation_voltages{ index_object }(1).axis ) ] );
                                excitation_voltages{ index_object } = discretizations.signal_matrix( excitation_voltages{ index_object }.axis(1), samples );
                            end

                        end

                    case 'discretizations.signal_matrix'

                        % ensure single signal matrix of correct size
                        if ~isscalar( excitation_voltages{ index_object } ) || ( numel( indices_active{ index_object } ) ~= excitation_voltages{ index_object }.N_signals )
                            errorStruct.message     = sprintf( 'excitation_voltages{ %d } must be a scalar and contain %d signals!', index_object, numel( indices_active{ index_object } ) );
                            errorStruct.identifier	= 'setting:SizeMismatch';
                            error( errorStruct );
                        end

                    otherwise

                        errorStruct.message     = sprintf( 'excitation_voltages{ %d } has to be discretizations.signal_matrix!', index_object );
                        errorStruct.identifier	= 'setting:NoSignalMatrix';
                        error( errorStruct );

                end % switch class( excitation_voltages{ index_object } )

                objects( index_object ).excitation_voltages = excitation_voltages{ index_object };

            end % for index_object = 1:numel( indices_active )

        end % function objects = setting_tx( indices_active, impulse_responses, excitation_voltages )

        %------------------------------------------------------------------
        % spectral discretization
        %------------------------------------------------------------------
        function objects_out = discretize( setting_tx, intervals_t, intervals_f )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( intervals_t, intervals_f );

            %--------------------------------------------------------------
            % 2.) compute transfer functions and excitation voltages
            %--------------------------------------------------------------
            % initialize cell arrays
            indices_active = cell( size( intervals_t ) );
            transfer_functions = cell( size( intervals_t ) );
            excitation_voltages = cell( size( intervals_t ) );

            % iterate intervals
            for index_object = 1:numel( intervals_t )

                indices_active{ index_object } = setting_tx.indices_active;
                transfer_functions{ index_object } = fourier_transform( setting_tx.impulse_responses, intervals_t( index_object ), intervals_f( index_object ) );
                excitation_voltages{ index_object } = fourier_coefficients( setting_tx.excitation_voltages, intervals_t( index_object ), intervals_f( index_object ) );

            end % for index_object = 1:numel( intervals_t )

            %--------------------------------------------------------------
            % 3.) create spectral discretizations of the recording settings
            %--------------------------------------------------------------
            objects_out = discretizations.spectral_points_tx( indices_active, transfer_functions, excitation_voltages );

        end % function objects_out = discretize( setting_tx, intervals_t, intervals_f )

	end % methods

end % classdef setting_tx < controls.setting

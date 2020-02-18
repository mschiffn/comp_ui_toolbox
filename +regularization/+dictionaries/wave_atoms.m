%
% superclass for all wave atom dictionaries
%
% author: Martin F. Schiffner
% date: 2020-01-25
% modified: 2020-02-17
%
classdef wave_atoms < regularization.dictionaries.dictionary

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties

        % independent properties
        type ( 1, 1 ) linear_transforms.wave_atoms.type { mustBeNonempty } = linear_transforms.wave_atoms.orthogonal

    end

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = wave_atoms( types )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure string array for types
            if ~isa( types, 'linear_transforms.wave_atoms.type' )
                errorStruct.message = 'types must be linear_transforms.wave_atoms.type!';
                errorStruct.identifier = 'wave_atoms:NoWaveAtomTypes';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) create wave atom dictionaries
            %--------------------------------------------------------------
            % constructor of superclass
            objects@regularization.dictionaries.dictionary( size( types ) );

            % iterate wave atom dictionaries
            for index_object = 1:numel( objects )

                % set independent properties
                objects( index_object ).type = types( index_object );

            end % for index_object = 1:numel( objects )

        end % function objects = wave_atoms( types )

        %------------------------------------------------------------------
        % string array (overload string method)
        %------------------------------------------------------------------
        function strs_out = string( dictionaries )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class regularization.dictionaries.wave_atoms
            if ~isa( dictionaries, 'regularization.dictionaries.wave_atoms' )
                errorStruct.message = 'dictionaries must be regularization.dictionaries.wave_atoms!';
                errorStruct.identifier = 'string:NoOptionsDictionaryWavelet';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) create string array
            %--------------------------------------------------------------
            % repeat string "wave atom"
            strs_out = repmat( "wave atom", size( dictionaries ) );

        end % function strs_out = string( dictionaries )

	end % methods

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (protected, hidden)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Access = protected, Hidden)

        %------------------------------------------------------------------
        % create linear transform (scalar)
        %------------------------------------------------------------------
        function LT = get_LT_scalar( dictionary, operator_born )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % calling function ensures class regularization.dictionaries.dictionary (scalar)
            % calling function ensures class scattering.operator_born (scalar)

            %--------------------------------------------------------------
            % 2.) create linear transform (scalar)
            %--------------------------------------------------------------
            % specify scales
            indicator_dimensions = operator_born.sequence.setup.FOV.shape.grid.N_points_axis > 1;
            N_dimensions = sum( indicator_dimensions );
            scales_finest = log2( operator_born.sequence.setup.FOV.shape.grid.N_points_axis( indicator_dimensions ) );

            % create linear transform
            LT = linear_transforms.wave_atom( dictionary.type, N_dimensions, scales_finest( 1 ) );

        end % function LT = get_LT_scalar( dictionary, operator_born )

	end % methods (Access = protected, Hidden)

end % classdef wave_atoms < regularization.dictionaries.dictionary

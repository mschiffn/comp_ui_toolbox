%
% superclass for all wave atom dictionary options
%
% author: Martin F. Schiffner
% date: 2020-01-25
% modified: 2020-01-30
%
classdef dictionary_wave_atoms < regularization.options.dictionary

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
        function objects = dictionary_wave_atoms( types )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure string array for types
            if ~isa( types, 'linear_transforms.wave_atoms.type' )
                errorStruct.message = 'types must be linear_transforms.wave_atoms.type!';
                errorStruct.identifier = 'dictionary_wave_atoms:NoWaveAtomTypes';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) create wave atom dictionary options
            %--------------------------------------------------------------
            % constructor of superclass
            objects@regularization.options.dictionary( size( types ) );

            % iterate wave atom dictionary options
            for index_object = 1:numel( objects )

                % set independent properties
                objects( index_object ).type = types( index_object );

            end % for index_object = 1:numel( objects )

        end % function objects = dictionary_wave_atoms( types )

        %------------------------------------------------------------------
        % string array (overload string method)
        %------------------------------------------------------------------
        function strs_out = string( dictionaries )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class regularization.options.dictionary_wave_atoms
            if ~isa( dictionaries, 'regularization.options.dictionary_wave_atoms' )
                errorStruct.message = 'dictionaries must be regularization.options.dictionary_wave_atoms!';
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

end % classdef dictionary_wave_atoms < regularization.options.dictionary

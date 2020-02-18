%
% superclass for all concatenated dictionaries
%
% author: Martin F. Schiffner
% date: 2020-01-12
% modified: 2020-02-17
%
classdef concatenation < regularization.dictionaries.dictionary

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties

        % independent properties
        dictionaries

        % dependent properties
        N_dictionaries ( 1, 1 ) double { mustBePositive, mustBeInteger, mustBeNonempty } = 2	% number of concatenated dictionaries

    end

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = concatenation( varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class regularization.dictionaries.dictionary
            indicator = cellfun( @( x ) ~isa( x, 'regularization.dictionaries.dictionary' ), varargin );
            if any( indicator( : ) )
                errorStruct.message = 'varargin must be regularization.dictionaries.dictionary!';
                errorStruct.identifier = 'concatenation:NoDictionaries';
                error( errorStruct );
            end

% TODO: reslove concatenation as input!

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( varargin{ : } );

            %--------------------------------------------------------------
            % 2.) create concatenated dictionaries
            %--------------------------------------------------------------
            % constructor of superclass
            objects@regularization.dictionaries.dictionary( size( varargin{ 1 } ) );

            % iterate concatenated dictionaries
            for index_object = 1:numel( objects )

                % set independent properties
                for index_arg = 1:nargin
                    objects( index_object ).dictionaries{ index_arg } = varargin{ index_arg }( index_object );
                end

                % set dependent properties
                objects( index_object ).N_dictionaries = numel( objects( index_object ).dictionaries );

            end % for index_object = 1:numel( objects )

        end % function objects = concatenation( varargin )

        %------------------------------------------------------------------
        % string array (overload string method)
        %------------------------------------------------------------------
        function strs_out = string( dictionaries_concatenated )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class regularization.dictionaries.concatenation
            if ~isa( dictionaries_concatenated, 'regularization.dictionaries.concatenation' )
                errorStruct.message = 'dictionaries_concatenated must be regularization.dictionaries.concatenation!';
                errorStruct.identifier = 'string:NoOptionsDictionaryConcatenated';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) create string array
            %--------------------------------------------------------------
            % repeat empty string
            strs_out = repmat( "", size( dictionaries_concatenated ) );

            % iterate concatenated dictionaries
            for index_object = 1:numel( dictionaries_concatenated )

                strs_out( index_object ) = sprintf( "concatenation (%s)", strjoin( cellfun( @( x ) string( x ), dictionaries_concatenated( index_object ).dictionaries ), '|' ) );

            end % for index_object = 1:numel( dictionaries_concatenated )

        end % function strs_out = string( dictionaries_concatenated )

	end % methods

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (protected, hidden)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Access = protected, Hidden)

        %------------------------------------------------------------------
        % create linear transform (scalar)
        %------------------------------------------------------------------
        function LT = get_LT_scalar( concatenation, operator_born )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % calling function ensures class regularization.dictionaries.dictionary (scalar)
            % calling function ensures class scattering.operator_born (scalar)

            %--------------------------------------------------------------
            % 2.) create linear transform (scalar)
            %--------------------------------------------------------------
            % specify cell array for linear transforms
            LT = cell( concatenation.N_dictionaries, 1 );

            % iterate concatenated dictionaries
            for index_dictionary = 1:concatenation.N_dictionaries

                % create linear transform for current dictionary
                LT{ index_dictionary } = get_LT_scalar( concatenation.dictionaries{ index_dictionary }, operator_born );

            end % for index_dictionary = 1:concatenation.N_dictionaries

            % create linear transform for current dictionary
            LT = linear_transforms.concatenations.vertical( LT{ : } );

        end % function LT = get_LT_scalar( concatenation, operator_born )

	end % methods (Access = protected, Hidden)

end % classdef concatenation < regularization.dictionaries.dictionary

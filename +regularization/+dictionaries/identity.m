%
% superclass for identity dictionaries
%
% author: Martin F. Schiffner
% date: 2019-12-28
% modified: 2020-02-17
%
classdef identity < regularization.dictionaries.dictionary

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = identity( varargin )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure nonempty size
            if nargin >= 1 && ~isempty( varargin{ 1 } )
                size = varargin{ 1 };
            else
                size = 1;
            end

            % superclass ensures row vector for size
            % superclass ensures nonempty positive integers

            %--------------------------------------------------------------
            % 2.) create identity dictionaries
            %--------------------------------------------------------------
            % constructor of superclass
            objects@regularization.dictionaries.dictionary( size );

        end % function objects = identity( varargin )

        %------------------------------------------------------------------
        % string array (overload string method)
        %------------------------------------------------------------------
        function strs_out = string( dictionaries_identity )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class regularization.dictionaries.identity
            if ~isa( dictionaries_identity, 'regularization.dictionaries.identity' )
                errorStruct.message = 'dictionaries_identity must be regularization.dictionaries.identity!';
                errorStruct.identifier = 'string:NoOptionsDictionaryIdentity';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) create string array
            %--------------------------------------------------------------
            % repeat string "identity"
            strs_out = repmat( "identity", size( dictionaries_identity ) );

        end % function strs_out = string( dictionaries_identity )

	end % methods

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (protected, hidden)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Access = protected, Hidden)

        %------------------------------------------------------------------
        % create linear transform (scalar)
        %------------------------------------------------------------------
        function LT = get_LT_scalar( ~, operator_born )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % calling function ensures class regularization.dictionaries.dictionary (scalar)
            % calling function ensures class scattering.operator_born (scalar)

            %--------------------------------------------------------------
            % 2.) create linear transform (scalar)
            %--------------------------------------------------------------
            % create linear transform
            LT = linear_transforms.identity( operator_born.sequence.setup.FOV.shape.grid.N_points );

        end % function LT = get_LT_scalar( ~, operator_born )

	end % methods (Access = protected, Hidden)

end % classdef identity < regularization.dictionaries.dictionary

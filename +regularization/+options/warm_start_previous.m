%
% superclass for all previous warm start options
%
% author: Martin F. Schiffner
% date: 2019-09-24
% modified: 2020-01-07
%
classdef warm_start_previous < regularization.options.warm_start

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = warm_start_previous( varargin )

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
            % 2.) create previous warm start options
            %--------------------------------------------------------------
            % constructor of superclass
            objects@regularization.options.warm_start( size );

        end % function objects = warm_start_previous( size )

        %------------------------------------------------------------------
        % display warm start options
        %------------------------------------------------------------------
        function str_out = show( warm_starts_previous )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class regularization.options.warm_start_previous
            if ~isa( warm_starts_previous, 'regularization.options.warm_start_previous' )
                errorStruct.message = 'warm_starts_previous must be regularization.options.warm_start_previous!';
                errorStruct.identifier = 'show:NoOptionsWarmStartPrevious';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) display options
            %--------------------------------------------------------------
            % specify cell array for str_out
            str_out = repmat( { 'previous' }, size( warm_starts_previous ) );

            % avoid cell array for single warm_starts_previous
            if isscalar( warm_starts_previous )
                str_out = str_out{ 1 };
            end

        end % function str_out = show( warm_starts_previous )

	end % methods

end % classdef warm_start_previous < regularization.options.warm_start

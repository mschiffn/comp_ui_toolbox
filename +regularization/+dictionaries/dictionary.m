%
% abstract superclass for dictionaries
%
% author: Martin F. Schiffner
% date: 2019-12-28
% modified: 2020-02-17
%
classdef (Abstract) dictionary < regularization.options.template

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = dictionary( size )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % superclass ensures row vector for size
            % superclass ensures nonempty positive integers for size

            %--------------------------------------------------------------
            % 2.) create dictionaries
            %--------------------------------------------------------------
            % constructor of superclass
            objects@regularization.options.template( size );

        end % function objects = dictionary( size )

        %------------------------------------------------------------------
        % create linear transforms
        %------------------------------------------------------------------
        function LTs = get_LTs( dictionaries, operators_born )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class regularization.dictionaries.dictionary
            if ~isa( dictionaries, 'regularization.dictionaries.dictionary' )
                errorStruct.message = 'dictionaries must be regularization.dictionaries.dictionary!';
                errorStruct.identifier = 'get_LTs:NoDictionaries';
                error( errorStruct );
            end

            % ensure class scattering.operator_born
            if ~isa( operators_born, 'scattering.operator_born' )
                errorStruct.message = 'operators_born must be scattering.operator_born!';
                errorStruct.identifier = 'get_LTs:NoOperatorsBorn';
                error( errorStruct );
            end

            % multiple dictionaries / single operators_born
            if ~isscalar( dictionaries ) && isscalar( operators_born )
                operators_born = repmat( operators_born, size( dictionaries ) );
            end

            % single dictionaries / multiple operators_born
            if isscalar( dictionaries ) && ~isscalar( operators_born )
                dictionaries = repmat( dictionaries, size( operators_born ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( dictionaries, operators_born );

            %--------------------------------------------------------------
            % 2.) create linear transforms
            %--------------------------------------------------------------
            % specify cell array for LTs
            LTs = cell( size( dictionaries ) );

            % iterate dictionaries
            for index_dictionary = 1:numel( dictionaries )

                % create linear transform (scalar)
                LTs{ index_dictionary } = get_LT_scalar( dictionaries( index_dictionary ), operators_born( index_dictionary ) );

            end % for index_dictionary = 1:numel( dictionaries )

            % avoid cell arrays for single dictionaries
            if isscalar( dictionaries )
                LTs = LTs{ 1 };
            end

        end % function LTs = get_LTs( dictionaries, operators_born )

	end % methods

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (Abstract, protected, hidden)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Abstract, Access = protected, Hidden)

        %------------------------------------------------------------------
        % create linear transform (scalar)
        %------------------------------------------------------------------
        LT = get_LT_scalar( dictionary, operator_born )

	end % methods (Abstract, Access = protected, Hidden)

end % classdef (Abstract) dictionary < regularization.options.template
